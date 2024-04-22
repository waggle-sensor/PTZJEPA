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
    batch_size = args['data']['rl_batch_size']
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
    TAU = args['optimization']['TAU']
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['rl_epochs']
    warmup = args['optimization']['rl_warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- PLATEAU
    patience = args['plateau']['rl_patience']
    threshold = args['plateau']['rl_threshold']

    # -- LOGGING
    folder = args['logging']['agent_folder']
    dream_folder = args['logging']['dream_folder']
    ownership_folder = args['logging']['ownership_folder']
    tag = args['logging']['write_tag']

    # -- MEMORY
    memory_models = args['memory']['rl_models']

    # -- DREAMER
    dream_length = args['dreamer']['dream_length']

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
    log_file = os.path.join(folder, model_ID, f'{tag}.csv')
    save_path = os.path.join(folder, model_ID, f'{tag}' + '-ep{epoch}.pth.tar')
    policy_latest_path = os.path.join(folder, model_ID, f'{tag}-policy_latest.pth.tar')
    target_latest_path = os.path.join(folder, model_ID, f'{tag}-target_latest.pth.tar')
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
    ipe = len(dataloader)*dream_length






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



    def save_checkpoint(epoch):
        policy_save_dict = {
            'predictor': policy_predictor.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'lr': lr
        }

        target_save_dict = {
            'predictor': target_predictor.state_dict(),
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'lr': lr
        }
        torch.save(policy_save_dict, policy_latest_path)
        torch.save(target_save_dict, target_latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(policy_save_dict, save_path.format(epoch=f'{epoch + 1}'))




    def tuple_of_tensors_to_tensor(tuple_of_tensors):
        return  torch.stack(list(tuple_of_tensors), dim=0)

    def optimize_model(transitions):
        GAMMA = 0.99
        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()
        # --

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = tuple_of_tensors_to_tensor(batch.state)
        position_batch = tuple_of_tensors_to_tensor(batch.position)
        action_batch = tuple_of_tensors_to_tensor(batch.action)
        next_state_batch = tuple_of_tensors_to_tensor(batch.next_state)
        next_position_batch = tuple_of_tensors_to_tensor(batch.next_position)
        reward_batch = tuple_of_tensors_to_tensor(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        action_batch=action_batch.round().to(torch.int64).view(-1, 1)
        state_action_values = torch.gather(policy_predictor(state_batch, position_batch), 1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = target_predictor(next_state_batch, next_position_batch).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = loss_fn(state_action_values, expected_state_action_values)

        # Backward & step
        loss.backward()
        optimizer.step()
        grad_stats = grad_logger(policy_predictor.named_parameters())
        optimizer.zero_grad()

        return (float(loss), _new_lr, _new_wd, grad_stats)




    # Compute Huber loss
    def loss_fn(state_action_values, expected_state_action_values):
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        return loss



    # -- Logging
    def log_stats(itr, epoch, loss, etime):
        csv_logger.log(epoch + 1, itr, loss, etime)
        if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
            logger.info('[%d, %5d] loss: %.3f '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '(%.1f ms)'
                        % (epoch + 1, itr,
                           loss_meter.avg,
                           _new_wd,
                           _new_lr,
                           torch.cuda.max_memory_allocated() / 1024.**2,
                           time_meter.avg))

            if grad_stats is not None:
                logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                            % (epoch + 1, itr,
                               grad_stats.first_layer,
                               grad_stats.last_layer,
                               grad_stats.min,
                               grad_stats.max))




    def detect_plateau(loss_values, patience=5, threshold=1e-4):
        """
        Detects plateauing behavior in a loss curve.

        Parameters:
            loss_values (list or numpy array): List or array containing the loss values over epochs.
            patience (int): Number of epochs with no improvement to wait before stopping.
            threshold (float): Threshold for the change in loss to be considered as plateauing.

        Returns:
            plateaued (bool): True if the loss has plateaued, False otherwise.
        """
        if len(loss_values) < patience + 1:
            return False  # Not enough data to detect plateauing

        recent_losses = loss_values[-patience:]
        mean_loss = np.mean(recent_losses)
        current_loss = loss_values[-1]

        if np.abs(current_loss - mean_loss) < threshold:
            return True  # Loss has plateaued
        else:
            return False  # Loss has not plateaued





    def change_ownership(folder):
        for subdir, dirs, files in os.walk(folder):
            os.chmod(subdir, 0o777)

            for File in files:
                os.chmod(os.path.join(subdir, File), 0o666)




    def prepare_data(dataloader, size):
        memory = ReplayMemory(size)
        for itr, episodes in enumerate(dataloader):
            for state_sequence, position_sequence, reward_sequence, action_sequence in zip(episodes['state_sequence'], episodes['possition_sequence'], episodes['reward_sequence'], episodes['action_sequence']):
                for step, (state, position) in enumerate(zip(state_sequence, position_sequence)):
                    if step < action_sequence.shape[0]:
                        # pick next action
                        action=action_sequence[step]
 
                        # next state and position
                        next_state=state_sequence[step+1]
                        next_position=position_sequence[step+1]

                        # reward
                        reward=reward_sequence[step]

                        # Store the transition in memory
                        memory.push(state, position, action, next_state, next_position, reward)

                        if len(memory) > 0.9*size:
                            return memory

        return memory






    # -- TRAINING LOOP
    #memory = ReplayMemory(100000)
    print('PREPARING DATA...')
    memory = prepare_data(dataloader, ipe)
    print('DONE!')
    loss_values = []
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr in range(len(memory)):
            # Perform one step of the optimization (on the policy network)
            #optimize_model()
            transitions = memory.sample(batch_size)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(optimize_model, arguments=transitions)
            loss_meter.update(loss)
            time_meter.update(etime)
            log_stats(itr, epoch, loss, etime)

            assert not np.isnan(loss), 'loss is nan'

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_predictor_state_dict = target_predictor.state_dict()
            policy_predictor_state_dict = policy_predictor.state_dict()
            with torch.no_grad():
                m = next(momentum_scheduler)
                for key in policy_predictor_state_dict:
                    #target_predictor_state_dict[key] = policy_predictor_state_dict[key]*m + target_predictor_state_dict[key]*(1-m)
                    target_predictor_state_dict[key] = policy_predictor_state_dict[key]*TAU + target_predictor_state_dict[key]*(1-TAU)
            target_predictor.load_state_dict(target_predictor_state_dict)

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        loss_values.append(loss_meter.avg)
        save_checkpoint(epoch+1)
        change_ownership(ownership_folder)
        if detect_plateau(loss_values, patience=patience, threshold=threshold):
            return False


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
