# this python script runs jepa

import time
import datetime
import pickle
import os
import glob
import shutil
import random
import copy
import logging
import yaml
import pprint
import torch
import torch.nn.functional as F
#import torch.nn.SmoothL1Loss as S1L

from PIL import Image 

from waggle.plugin import Plugin

from torch.utils.data import DataLoader

import numpy as np

from source.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)

from source.helper import (
    load_checkpoint,
    init_model,
    init_world_model,
    init_agent_model,
    init_opt)

from source.rl_helper import ReplayMemory, Transition

from source.transforms import make_transforms

#from source.datasets.ptz_dataset import PTZImageDataset
from source.datasets.dreams_dataset import DreamDataset

# --
#log_timings = True
log_freq = 10
checkpoint_freq = 50000000000000
# --


def control_ptz(args, params, logger=None, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = params['meta']['use_bfloat16']
    model_name = params['meta']['agent_model_name']
    load_model = params['meta']['load_checkpoint'] or resume_preempt
    r_file = params['meta']['read_checkpoint']
    pred_depth = params['meta']['pred_depth']
    pred_emb_dim = params['meta']['pred_emb_dim']
    camerabrand = params['meta']['camera_brand']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = params['data']['use_gaussian_blur']
    use_horizontal_flip = params['data']['use_horizontal_flip']
    use_color_distortion = params['data']['use_color_distortion']
    color_jitter = params['data']['color_jitter_strength']
    # --
    crop_size = params['data']['crop_size']
    crop_scale = params['data']['crop_scale']

    # -- MASK
    patch_size = params['mask']['patch_size']  # patch-size for model training
    # --

    # -- LOGGING
    folder = params['logging']['folder']
    agent_folder = params['logging']['agent_folder']
    dream_folder = params['logging']['dream_folder']
    ownership_folder = params['logging']['ownership_folder']
    tag = params['logging']['write_tag']

    # -- ACTIONS
    action_noop = params['action']['noop']
    action_short_left = params['action']['short']['left']
    action_short_right = params['action']['short']['right']
    action_short_left_up = params['action']['short']['left_up']
    action_short_right_up = params['action']['short']['right_up']
    action_short_left_down = params['action']['short']['left_down']
    action_short_right_down = params['action']['short']['right_down']
    action_short_up = params['action']['short']['up']
    action_short_down = params['action']['short']['down']
    action_short_zoom_in = params['action']['short']['zoom_in']
    action_short_zoom_out = params['action']['short']['zoom_out']

    action_long_left = params['action']['long']['left']
    action_long_right = params['action']['long']['right']
    action_long_up = params['action']['long']['up']
    action_long_down = params['action']['long']['down']
    action_long_zoom_in = params['action']['long']['zoom_in']
    action_long_zoom_out = params['action']['long']['zoom_out']

    action_jump_left = params['action']['jump']['left']
    action_jump_right = params['action']['jump']['right']
    action_jump_up = params['action']['jump']['up']
    action_jump_down = params['action']['jump']['down']

    actions={}
    actions[0]=action_noop
    actions[1]=action_short_left
    actions[2]=action_short_right
    actions[3]=action_short_left_up
    actions[4]=action_short_right_up
    actions[5]=action_short_left_down
    actions[6]=action_short_right_down
    actions[7]=action_short_up
    actions[8]=action_short_down
    actions[9]=action_short_zoom_in
    actions[10]=action_short_zoom_out
    actions[11]=action_long_left
    actions[12]=action_long_right
    actions[13]=action_long_up
    actions[14]=action_long_down
    actions[15]=action_long_zoom_in
    actions[16]=action_long_zoom_out
    actions[17]=action_jump_left
    actions[18]=action_jump_right
    actions[19]=action_jump_up
    actions[20]=action_jump_down

    num_actions=len(actions.keys())

    if not os.path.exists(folder) or not os.path.exists(agent_folder):
        print('No world models or agents to use the camera')
        return False


    world_models=[]
    for subdir in os.listdir(folder):
        world_models.append(subdir)

    if len(world_models)==0:
        print('No world models to use the camera')
        return False

    agents=[]
    for subdir in os.listdir(agent_folder):
        agents.append(subdir)

    if len(agents)==0:
        print('No agents to use the camera')
        return False

    world_model_ID=random.sample(world_models,1)[0]
    agent_ID=random.sample(agents,1)[0]

    # ----------------------------------------------------------------------- #
    #   Bring world model first
    # ----------------------------------------------------------------------- #

    print('world_model_ID: ', world_model_ID)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, world_model_ID, f'{tag}.csv')
    save_path = os.path.join(folder, world_model_ID, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, world_model_ID, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    print('log_file ', log_file)
    print('save_path ', save_path)
    print('latest_path ', latest_path)
    print('load_path ', load_path)

    # -- init world model
    target_encoder, _ = init_world_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)

    # -- make data transforms
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- load training checkpoint
    if load_model:
        _, _, target_encoder, _, _, _ = load_checkpoint(
            device=device,
            r_path=load_path,
            target_encoder=target_encoder)



    # ----------------------------------------------------------------------- #
    #   Bring agent model
    # ----------------------------------------------------------------------- #

    print('agent_ID: ', agent_ID)

    # -- log/checkpointing paths
    agent_log_file = os.path.join(agent_folder, agent_ID, f'{tag}.csv')
    agent_save_path = os.path.join(agent_folder, agent_ID, f'{tag}' + '-ep{epoch}.pth.tar')
    agent_target_latest_path = os.path.join(agent_folder, agent_ID, f'{tag}-target_latest.pth.tar')
    agent_target_load_path = None
    if load_model:
        agent_target_load_path = os.path.join(agent_folder, r_file) if r_file is not None else agent_target_latest_path

    print('agent_log_file ', agent_log_file)
    print('agent_save_path ', agent_save_path)
    print('agent_target_latest_path ', agent_target_latest_path)
    print('agent_target_load_path ', agent_target_load_path)

    # -- init agent model
    _, target_predictor = init_agent_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        num_actions=num_actions)

    for p in target_predictor.parameters():
        p.requires_grad = False

    # -- load training checkpoint
    if load_model:
        _, target_predictor, _, _, _, _ = load_checkpoint(
            device=device,
            r_path=agent_target_load_path,
            predictor=target_predictor)



    operate_ptz(args, actions, target_encoder, transform, target_predictor, device)




    return True








def change_ownership(folder):
    for subdir, dirs, files in os.walk(folder):
        os.chmod(subdir, 0o777)

        for File in files:
            os.chmod(os.path.join(subdir, File), 0o666)








def collect_positions(positions):
    directory=os.path.join('/persistence', 'collect_positions')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ct stores current time
    ct = str(datetime.datetime.now())

    afile = open(os.path.join(directory, 'positions_at_'+ct), 'wb')
    pickle.dump(positions, afile)
    afile.close()

    change_ownership(directory)



def collect_images(keepimages):
    directory = './collected_imgs'
    if not os.path.exists(directory):
        os.makedirs(directory)

    files = glob.glob('./imgs/*.jpg', recursive=True)
    for f in files:
        try:
            os.rename(f, os.path.join(directory, os.path.basename(f)))
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    if keepimages:
        src='./collected_imgs'
        dest=os.path.join('/persistence', 'collected_imgs')
        if not os.path.exists(dest):
            os.makedirs(dest)
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)


def set_random_position(camera, args):
    if args.camerabrand==0:
        pan_pos = np.random.randint(0, 360)
        tilt_pos = np.random.randint(-20, 90)
        zoom_pos = np.random.randint(1, 2)
    elif args.camerabrand==1:
        pan_pos = np.random.randint(-180, 180)
        tilt_pos = np.random.randint(-180, 180)
        zoom_pos = np.random.randint(100, 200)
    try:
        if args.camerabrand==0:
            camera.absolute_control(float(pan_pos), float(tilt_pos), float(zoom_pos))
        elif args.camerabrand==1:
            camera.absolute_move(float(pan_pos), float(tilt_pos), int(zoom_pos))
    except:
        with Plugin() as plugin:
            plugin.publish('cannot.set.camera.random.position', str(datetime.datetime.now()))

    time.sleep(1)





def set_relative_position(camera, args, pan, tilt, zoom):
    print('pan ', pan)
    print('tilt ', tilt)
    print('zoom ', zoom)
    try:
        if args.camerabrand==0:
            camera.relative_control(pan=pan, tilt=tilt, zoom=zoom)
        elif args.camerabrand==1:
            camera.relative_move(rpan=pan, rtilt=tilt, rzoom=zoom)
    except:
        with Plugin() as plugin:
            plugin.publish('cannot.set.camera.relative.position', str(datetime.datetime.now()))




def grab_position(camera, args):
    if args.camerabrand==0:
        position = camera.requesting_cameras_position_information()
    elif args.camerabrand==1:
        position = camera.get_ptz()

    pos_str = str(position[0]) + ',' + str(position[1]) + ',' + str(position[2]) + ' '

    return pos_str




def grab_image(camera, args):
    if args.camerabrand==0:
        position = camera.requesting_cameras_position_information()
    elif args.camerabrand==1:
        position = camera.get_ptz()

    pos_str = str(position[0]) + ',' + str(position[1]) + ',' + str(position[2]) + ' '
    # ct stores current time
    ct = str(datetime.datetime.now())
    try:
        camera.snap_shot('./imgs/' + pos_str + ct + '.jpg')
    except:
        with Plugin() as plugin:
            plugin.publish('cannot.capture.image.from.camera', str(datetime.datetime.now()))




def get_last_image(directory):
    last_timestamp=0
    last_imagepath=' '
    last_position=[0.0, 0.0, 0.0]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            position=f.split('_')[0].split('/')[-1].split(',')
            date=f.split('_')[-2].split('-')
            time=f.split('_')[-1].split('.')[0].split(':')
            year, month, day = int(date[0]), int(date[1]), int(date[2])
            hour, minute, second = int(time[0]), int(time[1]), int(time[2])
            pan, tilt, zoom = float(position[0]), float(position[1]), float(position[2])
            dt = datetime.datetime(year, month, day, hour, minute, second)
            if last_timestamp < dt.timestamp():
                last_timestamp = dt.timestamp()
                last_imagepath = f
                last_position = [pan, tilt, zoom]

    return Image.open(last_imagepath), torch.tensor(last_position)




def operate_ptz(args, actions, target_encoder, transform, target_predictor, device):
    if args.camerabrand==0:
        print('Importing Hanwha')
        from source import sunapi_control as sunapi_control
    elif args.camerabrand==1:
        print('Importing Axis')
        from source import vapix_control as sunapi_control
        #from source import onvif_control as sunapi_control
    else:
        print('Not known camera brand number: ', args.camerabrand)

    iterations = args.iterations
    number_of_commands = args.movements

    try:
        Camera1 = sunapi_control.CameraControl(args.cameraip, args.username, args.password)
    except:
        with Plugin() as plugin:
            plugin.publish('cannot.get.camera.from.ip', args.cameraip, timestamp=datetime.datetime.now())
            plugin.publish('cannot.get.camera.from.un', args.username, timestamp=datetime.datetime.now())
            plugin.publish('cannot.get.camera.from.pw', args.password, timestamp=datetime.datetime.now())
            

    if args.camerabrand==0:
        Camera1.absolute_control(1, 1, 1)
        time.sleep(1)
    elif args.camerabrand==1:
        Camera1.absolute_move(1, 1, 1)
        time.sleep(1)

    pan_modulation = 2
    tilt_modulation = 2
    zoom_modulation = 1

    pan_values = np.array([-5, -1, -0.1, 0, 0.1, 1, 5])
    pan_values = pan_values * pan_modulation
    tilt_values = np.array([-5, -1, -0.1, 0, 0.1, 1, 5])
    tilt_values = tilt_values * tilt_modulation
    if args.camerabrand==0:
        zoom_values = np.array([-0.2, -0.1, 0, 0.1, 0.2])
    elif args.camerabrand==1:
        zoom_values = 100*np.array([-2, -1, 0, 1, 2])

    zoom_values = zoom_values * zoom_modulation

    with Plugin() as plugin:
        plugin.publish('starting.new.image.collection.the.number.of.iterations.is', iterations)
        plugin.publish('the.number.of.images.recorded.by.iteration.is', number_of_commands)

    directory = './collected_imgs'
    if os.path.exists(directory):
        shutil.rmtree(directory)

    if os.path.exists('./imgs'):
        shutil.rmtree('./imgs')

    for iteration in range(iterations):
        with Plugin() as plugin:
            plugin.publish('iteration.number', iteration)

        os.mkdir('./imgs')
        PAN = np.random.choice(pan_values, number_of_commands)
        TILT = np.random.choice(tilt_values, number_of_commands)
        ZOOM = np.random.choice(zoom_values, number_of_commands)
        set_random_position(camera=Camera1, args=args)
        grab_image(camera=Camera1, args=args)

        positions = [grab_position(camera=Camera1, args=args)]
        for command in range(number_of_commands):
            image, position = get_last_image('./imgs')
            image = transform(image)
            image = image.unsqueeze(0)
            position_batch = position.unsqueeze(0).to(device)
            state_batch = target_encoder(image.to(device))
            with torch.no_grad():
                #next_state_values = target_predictor(state_batch, position_batch)
                max_next_state_indices = target_predictor(state_batch, position_batch).max(1).indices.item()
                #next_state_values = target_predictor(state_batch, position_batch).max(1).values
                next_state_values = target_predictor(state_batch, position_batch)
                # Apply softmax to convert to probabilities
                probs = F.softmax(next_state_values, dim=1)
                # Sample indices based on the probability distribution
                num_samples = 1  # Adjust as needed
                sampled_indices = torch.multinomial(probs.squeeze(), num_samples, replacement=True)

            print('next_state_values: ', next_state_values)
            print('probs: ', probs)
            #print('max_next_state_indices: ', max_next_state_indices)
            if torch.rand([1]).item() > 0.9:
                print('Sampled action')
                print('sampled_indices: ', sampled_indices.item())
                next_action = actions[sampled_indices.item()]
            else:
                print('Rewarded action')
                print('max_next_state_indices: ', max_next_state_indices)
                next_action = actions[max_next_state_indices]
            print('next_action: ', next_action)

            pan_modulation = 2
            tilt_modulation = 2
            if args.camerabrand==0:
                zoom_modulation = 1
            elif args.camerabrand==1:
                zoom_modulation = 100

            pan=next_action[0]*pan_modulation
            tilt=next_action[1]*tilt_modulation
            zoom=next_action[2]*zoom_modulation

            #set_random_position(camera=Camera1, args=args) ## I have to simply replace it with the decisions taken by the agent
            set_relative_position(camera=Camera1, args=args,
                                  pan=pan,
                                  tilt=tilt,
                                  zoom=zoom)
            grab_image(camera=Camera1, args=args)
            positions.append(grab_position(camera=Camera1, args=args))
        #for (pan, tilt, zoom) in zip(PAN, TILT, ZOOM):
            #try:
                #if args.camerabrand==0:
                    #Camera1.relative_control(pan=pan, tilt=tilt, zoom=zoom)
                #elif args.camerabrand==1:
                    #Camera1.relative_move(rpan=pan, rtilt=tilt, rzoom=zoom)
            #except:
                #with Plugin() as plugin:
                    #plugin.publish('cannot.set.camera.relative.position', str(datetime.datetime.now()))

            #grab_image(camera=Camera1, args=args)

        #publish_images()
        collect_images(args.keepimages)
        os.rmdir('./imgs')

        if args.trackpositions:
            collect_positions(positions)
















def run(args, fname, mode):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info('called-params %s', fname)

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        logger.info('loading params...')
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    if mode=='navigate_env':
        return control_ptz(args, params, logger=logger)
    else:
        raise ValueError(f"Unexpected mode {mode}")
