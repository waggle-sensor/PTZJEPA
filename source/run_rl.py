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
    init_world_model,
    init_opt)
from source.transforms import make_transforms

from source.datasets.ptz_dataset import PTZImageDataset

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
    ownership_folder = args['logging']['ownership_folder']
    tag = args['logging']['write_tag']

    # -- MEMORY
    memory_models = args['memory']['models']

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
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    print('log_file ', log_file)
    print('save_path ', save_path)
    print('latest_path ', latest_path)
    print('load_path ', load_path)





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
