# this python script runs jepa

from pathlib import Path
import time
import datetime
import os
import shutil
import random
import logging
import yaml
import pprint
import torch
import torch.nn.functional as F
#import torch.nn.SmoothL1Loss as S1L

from PIL import Image

from waggle.plugin import Plugin


import numpy as np

from source.datasets.ptz_dataset import get_position_datetime_from_labels
from source.prepare_dataset import (
    collect_commands,
    collect_embeds,
    collect_images,
    collect_positions,
    grab_image, grab_position,
    set_random_position,
    set_relative_position,
    get_dirs,
    verify_image
)


from source.helper import (
    load_checkpoint,
    init_world_model,
    init_agent_model
)


from source.transforms import make_transforms

#from source.datasets.ptz_dataset import PTZImageDataset

# --
#log_timings = True
log_freq = 10
checkpoint_freq = 50000000000000
# --
logger = logging.getLogger(__name__)


def control_ptz(args, params, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = params['meta']['use_bfloat16']
    model_arch = params['meta']['agent_model_arch']
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

    # Need to determine the id to use here
    world_model_ID=random.sample(world_models,1)[0]
    agent_ID=random.sample(agents,1)[0]

    # ----------------------------------------------------------------------- #
    #   Bring world model first
    # ----------------------------------------------------------------------- #

    print('world_model_ID: ', world_model_ID)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, world_model_ID, f'{tag}.csv')
    save_path = os.path.join(folder, world_model_ID, f'{tag}' + '-ep{epoch}.pt')
    latest_path = os.path.join(folder, world_model_ID, f'{tag}-latest.pt')
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
        model_arch=model_arch)

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
    agent_save_path = os.path.join(agent_folder, agent_ID, f'{tag}' + '-ep{epoch}.pt')
    agent_target_latest_path = os.path.join(agent_folder, agent_ID, f'{tag}-target_latest.pt')
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
        model_arch=model_arch,
        num_actions=num_actions)

    for p in target_predictor.parameters():
        p.requires_grad = False

    # -- load training checkpoint
    if load_model:
        _, target_predictor, _, _, _, _ = load_checkpoint(
            device=device,
            r_path=agent_target_load_path,
            predictor=target_predictor)



    operate_ptz_with_agent(args, actions, target_encoder, transform, target_predictor, device)
    return True


def get_last_image(directory):
    directory = Path(directory)
    all_files = [fp.stem for fp in directory.glob('*.jpg')]
    arr_pos, arr_datetime = get_position_datetime_from_labels(all_files)
    idx = np.argmax(arr_datetime)
    return Image.open(directory / f"{all_files[idx]}.jpg"), torch.tensor(arr_pos[idx])


def operate_ptz_with_agent(args, actions, target_encoder, transform, target_predictor, device):
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

    persis_dir, coll_dir, tmp_dir = get_dirs()
    if coll_dir.exists():
        shutil.rmtree(coll_dir)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    for iteration in range(iterations):
        with Plugin() as plugin:
            plugin.publish('iteration.number', iteration)

        tmp_dir.mkdir(exist_ok=True)
        # Get first random image as a starting point
        set_random_position(camera=Camera1, args=args)
        grab_image(camera=Camera1, args=args)

        positions = [grab_position(camera=Camera1, args=args)]
        cmds = []
        embeds = []
        for command in range(number_of_commands):
            image, position = get_last_image(tmp_dir)
            image = transform(image)
            image = image.unsqueeze(0)
            position_batch = position.unsqueeze(0).to(device, dtype=torch.float32)
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
            # Make sure the image is captured before moving on
            count = 0
            while count < 10:
                img_path = grab_image(camera=Camera1, args=args)
                if img_path and verify_image(img_path):
                    break
                count += 1
            if count == 10 and not img_path:
                logger.warning("Failed to grab image after 10 attempts, skip this command")
                if os.path.exists(img_path):
                    os.remove(img_path)
                continue
            positions.append(grab_position(camera=Camera1, args=args))
            cmds.append(",".join((pan, tilt, zoom)))
            embeds.append(state_batch.detach().cpu())

        #publish_images()
        collect_images(args.keepimages)
        shutil.rmtree(tmp_dir)
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        if args.trackpositions or args.track_all:
            collect_positions(positions, cur_time)
            collect_commands(cmds, cur_time)
        
        if args.track_all:
            collect_embeds(embeds, cur_time)


def run(args, fname, mode):

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
        return control_ptz(args, params)
    else:
        raise ValueError(f"Unexpected mode {mode}")
