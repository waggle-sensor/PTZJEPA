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
#import torch.nn.SmoothL1Loss as S1L

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


def operate_ptz(args, params, logger=None, resume_preempt=False):
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
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    crop_size = params['data']['crop_size']
    # --

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

    print('world_model_ID: ', world_model_ID)
    print('agent_ID: ', agent_ID)









    return True























def run(args, fname, mode):
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

    if mode=='navigate_env':
        return operate_ptz(args, params, logger=logger)
    else:
        print(f"Unexpected mode {mode}")
        raise
