# this python script runs jepa

import os
import shutil
import random
import copy
import logging
import yaml
import pprint
import torch
import torch.nn.functional as F

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


def agent_model(args, logger=None, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['agent_model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- PLATEAU
    patience = args['plateau']['patience']
    threshold = args['plateau']['threshold']

    # -- LOGGING
    folder = args['logging']['agent_folder']
    dream_folder = args['logging']['dream_folder']
    ownership_folder = args['logging']['ownership_folder']
    tag = args['logging']['write_tag']

    # -- MEMORY
    memory_models = args['memory']['models']

    # -- ACTIONS
    action_short_left = args['action']['short']['left']
    action_short_right = args['action']['short']['right']
    action_short_left_up = args['action']['short']['left_up']
    action_short_right_up = args['action']['short']['right_up']
    action_short_left_down = args['action']['short']['left_down']
    action_short_right_down = args['action']['short']['right_down']
    action_short_up = args['action']['short']['up']
    action_short_down = args['action']['short']['down']
    action_short_zoom_in = args['action']['short']['zoom_in']
    action_short_zoom_out = args['action']['short']['zoom_out']

    action_long_left = args['action']['long']['left']
    action_long_right = args['action']['long']['right']
    action_long_up = args['action']['long']['up']
    action_long_down = args['action']['long']['down']
    action_long_zoom_in = args['action']['long']['zoom_in']
    action_long_zoom_out = args['action']['long']['zoom_out']

    actions={}
    actions[0]=action_short_left
    actions[1]=action_short_right
    actions[2]=action_short_left_up
    actions[3]=action_short_right_up
    actions[4]=action_short_left_down
    actions[5]=action_short_right_down
    actions[6]=action_short_up
    actions[7]=action_short_down
    actions[8]=action_short_zoom_in
    actions[9]=action_short_zoom_out
    actions[10]=action_long_left
    actions[11]=action_long_right
    actions[12]=action_long_up
    actions[13]=action_long_down
    actions[14]=action_long_zoom_in
    actions[15]=action_long_zoom_out

    num_actions=len(actions.keys())

    if not os.path.exists(folder):
        os.makedirs(folder)

    model_ID='model_'+str(torch.randint(memory_models, (1,)).item())
    if not os.path.exists(os.path.join(folder, model_ID)):
        os.makedirs(os.path.join(folder, model_ID))

    dump = os.path.join(folder, model_ID, 'params-agent.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #




    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    policy_latest_path = os.path.join(folder, f'{tag}-policy_latest.pth.tar')
    target_latest_path = os.path.join(folder, f'{tag}-target_latest.pth.tar')
    policy_load_path = None
    target_load_path = None
    if load_model:
        policy_load_path = os.path.join(folder, r_file) if r_file is not None else policy_latest_path
        target_load_path = os.path.join(folder, r_file) if r_file is not None else target_latest_path

    print('log_file ', log_file)
    print('save_path ', save_path)
    print('policy_latest_path ', policy_latest_path)
    print('target_latest_path ', target_latest_path)
    print('policy_load_path ', policy_load_path)
    print('target_load_path ', target_load_path)

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init world model
    encoder, policy_predictor = init_agent_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        num_actions=num_actions)
    target_predictor = copy.deepcopy(policy_predictor)


    # -- init data-loader
    data = DreamDataset(dream_folder)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    ipe = len(dataloader)






    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=policy_predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)






    for p in target_predictor.parameters():
        p.requires_grad = False


    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))




    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        _, policy_predictor, _, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=policy_load_path,
            predictor=policy_predictor,
            opt=optimizer,
            scaler=scaler)

        _, target_predictor, _, _, _, _ = load_checkpoint(
            device=device,
            r_path=target_load_path,
            predictor=target_predictor)

        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)



    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)








    # -- TRAINING LOOP
    memory = ReplayMemory(10000)
    TAU=0.5
    loss_values = []
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, episodes in enumerate(dataloader):
            for state_sequence, position_sequence, reward_sequence, action_sequence in zip(episodes['state_sequence'], episodes['possition_sequence'], episodes['reward_sequence'], episodes['action_sequence']):
                for step, (state, position) in enumerate(zip(state_sequence, position_sequence)):
                    print(step)

                    if step < action_sequence.shape[0]:
                        # pick next action
                        action=action_sequence[step]
                        
                        # next state and position
                        next_state=state_sequence[step+1]
                        next_position=position_sequence[step+1]

                        # reward
                        reward=reward_sequence[step]

                        # Store the transition in memory
                        memory.push(state, next_state, action, next_state, next_position, reward)

                        # Perform one step of the optimization (on the policy network)
                        optimize_model()

                        # Soft update of the target network's weights
                        # θ′ ← τ θ + (1 −τ )θ′
                        target_predictor_state_dict = target_predictor.state_dict()
                        policy_predictor_state_dict = policy_predictor.state_dict()
                        for key in policy_predictor_state_dict:
                            target_predictor_state_dict[key] = policy_predictor_state_dict[key]*TAU + target_predictor_state_dict[key]*(1-TAU)
                        target_predictor.load_state_dict(target_predictor_state_dict)



    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('HHHHHHHHHHHHHHHHHEEEEEEEEEEEEEEYYYYYYYYYYYYYYYYYYY______IIIIIIIAMMMMMMMMMHHHHHHHHHHHHHHEEEEEEEERRREE')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')







    return True























def run(fname, mode):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        logger.info('loading params...')
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    if mode=='train_agent':
        return agent_model(params, logger=logger)
    else:
        print(f"Unexpected mode {mode}")
        raise
