# this python script runs jepa

import gc
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

def train(args, logger=None, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    camera_brand = args['meta']['camera_brand'] #TODO I have to fix it!!!!!!!!!! I have to include the arguments of main together with the arguments from the yalm file
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
    global_batch_size = args['data']['global_batch_size']
    batch_size = args['data']['batch_size']
    accumulation_steps = int(global_batch_size / batch_size)
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
    folder = args['logging']['folder']
    ownership_folder = args['logging']['ownership_folder']
    tag = args['logging']['write_tag']


    def change_ownership(folder):
        for subdir, dirs, files in os.walk(folder):
            os.chmod(subdir, 0o777)

            for File in files:
                os.chmod(os.path.join(subdir, File), 0o666)



    if not os.path.exists(folder):
        os.makedirs(folder)
        change_ownership(folder)

    dump = os.path.join(folder, 'params-ijepa.yaml')
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

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'lr'),
                           ('%.5f', 'wd'),
                           #('%.5f', 'mask-A'),
                           #('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    ## -- make csv_logger
    #csv_logger = CSVLogger(log_file,
    #                       ('%d', 'epoch'),
    #                       ('%d', 'itr'),
    #                       ('%.5f', 'loss'),
    #                       ('%.5f', 'mask-A'),
    #                       ('%.5f', 'mask-B'),
    #                       ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)



    # -- make data transforms
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)


    # -- init data-loader
    data = PTZImageDataset('./labels', './collected_imgs', transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    ipe = len(dataloader)



    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
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


    for p in target_encoder.parameters():
        p.requires_grad = False


    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))


    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)


    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'lr': lr
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))


    def get_position_from_label(labels):
        possitions = []
        for label in labels:
            poss = label.split('_')[0].split(',')
            possitions.append([float(poss[0]), float(poss[1]), float(poss[2])])

        return torch.tensor(possitions)


    def arrage_inputs(images, possitions):
        context_imgs = []
        target_imgs = []
        context_poss = []
        target_poss = []
        for context_idx in range(possitions.shape[0]):
            for target_idx in range(possitions.shape[0]):
                context_imgs.append(images[context_idx].unsqueeze(0))
                context_poss.append(possitions[context_idx].unsqueeze(0))
                target_imgs.append(images[target_idx].unsqueeze(0))
                target_poss.append(possitions[target_idx].unsqueeze(0))


        auxiliary = torch.Tensor(len(context_imgs), context_imgs[0].shape[1],
                                                    context_imgs[0].shape[2],
                                                    context_imgs[0].shape[3]).to(device, non_blocking=True)
        torch.cat(context_imgs, out=auxiliary)
        auxiliary = auxiliary.squeeze(1)
        context_imgs = auxiliary.detach().clone()

        auxiliary = torch.Tensor(len(context_poss), context_poss[0].shape[1]).to(device, non_blocking=True)
        torch.cat(context_poss, out=auxiliary)
        auxiliary = auxiliary.squeeze(1)
        context_poss = auxiliary.detach().clone()


        auxiliary = torch.Tensor(len(target_imgs), target_imgs[0].shape[1],
                                                   target_imgs[0].shape[2],
                                                   target_imgs[0].shape[3]).to(device, non_blocking=True)
        torch.cat(target_imgs, out=auxiliary)
        auxiliary = auxiliary.squeeze(1)
        target_imgs = auxiliary.detach().clone()

        auxiliary = torch.Tensor(len(target_poss), target_poss[0].shape[1]).to(device, non_blocking=True)
        torch.cat(target_poss, out=auxiliary)
        auxiliary = auxiliary.squeeze(1)
        target_poss = auxiliary.detach().clone()

        return context_imgs.to(device), context_poss.to(device), target_imgs.to(device), target_poss.to(device)
 

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



    def get_random_position():
        if camera_brand==0:
            pan_pos = np.random.randint(0, 360)
            tilt_pos = np.random.randint(-90, 90)
            zoom_pos = np.random.randint(1, 2)
        elif camera_brand==1:
            pan_pos = np.random.randint(-180, 180)
            tilt_pos = np.random.randint(-180, 180)
            zoom_pos = np.random.randint(100, 200)

        return pan_pos, tilt_pos, zoom_pos




    def train_step(inputs):
        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()
        # --

        # Step 1. Forward
        h = forward_target(inputs[2])
        z = forward_context(inputs[0], inputs[1], inputs[3])
        loss = loss_fn(z, h)
        # Divide loss by accumulation steps
        loss = loss / accumulation_steps



        # Step 2. Backward & step
        loss.backward()
        ## Gradient Value Clipping
        #torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.1)
        #torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=0.1)
        optimizer.step()
        grad_stats = grad_logger(encoder.named_parameters())
        optimizer.zero_grad()


        # Step 3. momentum update of target encoder
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return (float(loss), _new_lr, _new_wd, grad_stats)



    def acumulate_train_step(inputs):
        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()
        # --

        # Step 1. Forward
        h = forward_target(inputs[2])
        z = forward_context(inputs[0], inputs[1], inputs[3])
        loss = loss_fn(z, h)
        # Divide loss by accumulation steps
        loss = loss / accumulation_steps

        # Step 2. Backward & step
        loss.backward()
        #optimizer.step()
        grad_stats = grad_logger(encoder.named_parameters())
        #optimizer.zero_grad()


        # Step 3. momentum update of target encoder
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return (float(loss), _new_lr, _new_wd, grad_stats)



    def forward_target(images):
        with torch.no_grad():
            h = target_encoder(images)
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            return h

    def forward_context(images, possition1, possition2):
        # Change allocentric position
        change_allocentric_position(possition1, possition2)
        z = encoder(images)
        z = predictor(z, possition1, possition2) 
        return z


    def loss_fn(z, h):
        loss = F.smooth_l1_loss(z, h)
        return loss


    # -- Logging
    #def log_stats(itr, epoch, loss, etime):
    def log_stats(itr, epoch, loss, lr, wd, etime):
        csv_logger.log(epoch + 1, itr, loss, lr, wd, etime)
        #csv_logger.log(epoch + 1, itr, loss, etime)
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


    # Change allocentric position
    def change_allocentric_position(possition1, possition2):
        pan, tilt, _ = get_random_position()
        possition1[:,0] = possition1[:,0] - pan
        possition1[:,1] = possition1[:,1] - tilt
        possition2[:,0] = possition2[:,0] - pan
        possition2[:,1] = possition2[:,1] - tilt




    # -- TRAINING LOOP
    loss_values = []
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (imgs, labls) in enumerate(dataloader):
            poss = get_position_from_label(labls)
            imgs = imgs.to(device, non_blocking=True)
            poss = poss.to(device, non_blocking=True)
            
            context_imgs, context_poss, target_imgs, target_poss = arrage_inputs(imgs, poss)

            if itr%int(global_batch_size/batch_size)==0:
                (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step, arguments=[context_imgs, context_poss, target_imgs, target_poss])
            else:
                (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(acumulate_train_step, arguments=[context_imgs, context_poss, target_imgs, target_poss])

            #(loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step, arguments=[context_imgs, context_poss, target_imgs, target_poss])
            loss_meter.update(loss)
            time_meter.update(etime)
            log_stats(itr, epoch, loss, _new_lr, _new_wd, etime)
            #log_stats(itr, epoch, loss, etime)

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        loss_values.append(loss_meter.avg)
        save_checkpoint(epoch+1)
        change_ownership(ownership_folder)
        if detect_plateau(loss_values, patience=patience, threshold=threshold):
            return False

    
    return True

















def world_model(args, logger=None, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    camera_brand = args['meta']['camera_brand']
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
    global_batch_size = args['data']['global_batch_size']
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
    patience = args['plateau']['wm_patience']
    threshold = args['plateau']['wm_threshold']

    # -- LOGGING
    folder = args['logging']['folder']
    ownership_folder = args['logging']['ownership_folder']
    tag = args['logging']['write_tag']

    # -- MEMORY
    memory_models = args['memory']['models']



    def change_ownership(folder):
        for subdir, dirs, files in os.walk(folder):
            os.chmod(subdir, 0o777)

            for File in files:
                os.chmod(os.path.join(subdir, File), 0o666)



    if not os.path.exists(folder):
        os.makedirs(folder)
        change_ownership(folder)

    model_ID='model_'+str(torch.randint(memory_models, (1,)).item())
    if not os.path.exists(os.path.join(folder, model_ID)):
        os.makedirs(os.path.join(folder, model_ID))
        change_ownership(os.path.join(folder, model_ID))

    dump = os.path.join(folder, model_ID, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #
    
    # -- log/checkpointing paths
    log_file = os.path.join(folder, model_ID, f'{tag}.csv')
    save_path = os.path.join(folder, model_ID, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, model_ID, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    print('log_file ', log_file)
    print('save_path ', save_path)
    print('latest_path ', latest_path)
    print('load_path ', load_path)

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'lr'),
                           ('%.5f', 'wd'),
                           #('%.5f', 'mask-A'),
                           #('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init world model
    encoder, predictor = init_world_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)



    # -- make data transforms
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)




    # -- init data-loader
    data = PTZImageDataset('./labels', './collected_imgs', transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    ipe = len(dataloader)
    #ipe = int(len(dataloader)*(batch_size/global_batch_size))






    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
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



    #for p in target_encoder.parameters():
    #    p.requires_grad = False


    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))



    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)




    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'lr': lr
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))


    def get_position_from_label(labels):
        possitions = []
        for label in labels:
            poss = label.split('_')[0].split(',')
            possitions.append([float(poss[0]), float(poss[1]), float(poss[2])])

        return torch.tensor(possitions)


    def arrage_inputs(images, possitions):
        context_imgs = []
        target_imgs = []
        context_poss = []
        target_poss = []
        for context_idx in range(possitions.shape[0]):
            for target_idx in range(possitions.shape[0]):
                context_imgs.append(images[context_idx].unsqueeze(0))
                context_poss.append(possitions[context_idx].unsqueeze(0))
                target_imgs.append(images[target_idx].unsqueeze(0))
                target_poss.append(possitions[target_idx].unsqueeze(0))


        auxiliary = torch.Tensor(len(context_imgs), context_imgs[0].shape[1],
                                                    context_imgs[0].shape[2],
                                                    context_imgs[0].shape[3]).to(device, non_blocking=True)
        torch.cat(context_imgs, out=auxiliary)
        auxiliary = auxiliary.squeeze(1)
        context_imgs = auxiliary.detach().clone()

        auxiliary = torch.Tensor(len(context_poss), context_poss[0].shape[1]).to(device, non_blocking=True)
        torch.cat(context_poss, out=auxiliary)
        auxiliary = auxiliary.squeeze(1)
        context_poss = auxiliary.detach().clone()


        auxiliary = torch.Tensor(len(target_imgs), target_imgs[0].shape[1],
                                                   target_imgs[0].shape[2],
                                                   target_imgs[0].shape[3]).to(device, non_blocking=True)
        torch.cat(target_imgs, out=auxiliary)
        auxiliary = auxiliary.squeeze(1)
        target_imgs = auxiliary.detach().clone()

        auxiliary = torch.Tensor(len(target_poss), target_poss[0].shape[1]).to(device, non_blocking=True)
        torch.cat(target_poss, out=auxiliary)
        auxiliary = auxiliary.squeeze(1)
        target_poss = auxiliary.detach().clone()

        return context_imgs.to(device), context_poss.to(device), target_imgs.to(device), target_poss.to(device)
 



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










    def get_random_position():
        if camera_brand==0:
            pan_pos = np.random.randint(0, 360)
            tilt_pos = np.random.randint(-90, 90)
            zoom_pos = np.random.randint(1, 2)
        elif camera_brand==1:
            pan_pos = np.random.randint(-180, 180)
            tilt_pos = np.random.randint(-180, 180)
            zoom_pos = np.random.randint(100, 200)

        return pan_pos, tilt_pos, zoom_pos







    def train_step(inputs):
        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()
        # --

        # Step 1. Auxiliary Forward
        h = forward_target(inputs[2])
        with torch.no_grad():
            z, r = forward_context(inputs[0], inputs[1], inputs[3])
        auxiliary_loss = auxiliary_loss_fn(z, h)

        # Step 2. Auxiliary Backward
        auxiliary_loss.backward()
        grads = []
        for param in target_encoder.parameters():
            if param.grad != None:
                grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        g = grads.abs().sum()
        target_encoder.zero_grad()

        # Step 3. Forward
        with torch.no_grad():
            h = forward_target(inputs[2])
        z, r = forward_context(inputs[0], inputs[1], inputs[3])
        loss = loss_fn(z, r, h, g)

        # Step 4. Backward & step
        loss.backward()
        optimizer.step()
        grad_stats = grad_logger(encoder.named_parameters())
        optimizer.zero_grad()


        # Step 5. momentum update of target encoder
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.detach().data.mul_(m).add_((1.-m) * param_q.detach().data)
                #param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return (float(loss), _new_lr, _new_wd, grad_stats)





    def acumulate_train_step(inputs):
        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()
        # --

        # Step 1. Auxiliary Forward
        h = forward_target(inputs[2])
        with torch.no_grad():
            z, r = forward_context(inputs[0], inputs[1], inputs[3])
        auxiliary_loss = auxiliary_loss_fn(z, h)

        # Step 2. Auxiliary Backward
        auxiliary_loss.backward()
        grads = []
        for param in target_encoder.parameters():
            if param.grad != None:
                grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        g = grads.abs().sum()
        target_encoder.zero_grad()

        # Step 3. Forward
        with torch.no_grad():
            h = forward_target(inputs[2])
        z, r = forward_context(inputs[0], inputs[1], inputs[3])
        loss = loss_fn(z, r, h, g)

        # Step 4. Backward & step
        loss.backward()
        #optimizer.step()
        grad_stats = grad_logger(encoder.named_parameters())
        #optimizer.zero_grad()


        # Step 5. momentum update of target encoder
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.detach().data.mul_(m).add_((1.-m) * param_q.detach().data)
                #param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return (float(loss), _new_lr, _new_wd, grad_stats)









    def forward_target(images):
        h = target_encoder(images)
        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
        return h

    def forward_context(images, possition1, possition2):
        # Change allocentric position
        change_allocentric_position(possition1, possition2)
        z = encoder(images)
        z, r = predictor(z, possition1, possition2) 
        return z, r


    def auxiliary_loss_fn(z, h):
        loss = F.smooth_l1_loss(z, h)
        return loss

    def loss_fn(z, r, h, g):
        loss1 = F.smooth_l1_loss(z, h)
        loss2 = F.smooth_l1_loss(r, g.repeat(r.shape))# * 1e-4
        loss = loss1 + loss2
        return loss


    # -- Logging
    def log_stats(itr, epoch, loss, lr, wd, etime):
        csv_logger.log(epoch + 1, itr, loss, lr, wd, etime)
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

    # Change allocentric position
    def change_allocentric_position(possition1, possition2):
        pan, tilt, _ = get_random_position()
        possition1[:,0] = possition1[:,0] - pan
        possition1[:,1] = possition1[:,1] - tilt
        possition2[:,0] = possition2[:,0] - pan
        possition2[:,1] = possition2[:,1] - tilt




    # -- TRAINING LOOP
    change_ownership(os.path.join(folder, model_ID))
    loss_values = []
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (imgs, labls) in enumerate(dataloader):
            poss = get_position_from_label(labls)
            imgs = imgs.to(device, non_blocking=True)
            poss = poss.to(device, non_blocking=True)
            
            context_imgs, context_poss, target_imgs, target_poss = arrage_inputs(imgs, poss)

            if itr%int(global_batch_size/batch_size)==0:
                (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step, arguments=[context_imgs, context_poss, target_imgs, target_poss])
            else:
                #loss, etime = gpu_timer(acumulate_train_step, arguments=[context_imgs, context_poss, target_imgs, target_poss])
                (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(acumulate_train_step, arguments=[context_imgs, context_poss, target_imgs, target_poss])

            loss_meter.update(loss)
            time_meter.update(etime)
            log_stats(itr, epoch, loss, _new_lr, _new_wd, etime)

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        loss_values.append(loss_meter.avg)
        save_checkpoint(epoch+1)
        change_ownership(ownership_folder)
        if detect_plateau(loss_values, patience=patience, threshold=threshold):
            encoder.cpu()
            predictor.cpu()
            target_encoder.cpu()
            del encoder, predictor, target_encoder, dataloader, data
            gc.collect()
            torch.cuda.empty_cache()
            return False


    if start_epoch == num_epochs:
        PATH, FILE = os.path.split(latest_path)
        shutil.rmtree(PATH)

    return True




















def dreamer(args, logger=None, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    camera_brand = args['meta']['camera_brand']
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

    # -- DREAMER
    number_of_dreams = args['dreamer']['number_of_dreams']
    dream_length = args['dreamer']['dream_length']

    # -- LOGGING
    folder = args['logging']['folder']
    dream_folder = args['logging']['dream_folder']
    ownership_folder = args['logging']['ownership_folder']
    tag = args['logging']['write_tag']

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

    # -- MEMORY
    memory_dreams = args['memory']['dreams']



    def change_ownership(folder):
        for subdir, dirs, files in os.walk(folder):
            os.chmod(subdir, 0o777)

            for File in files:
                os.chmod(os.path.join(subdir, File), 0o666)


    
    if not os.path.exists(folder):
        os.makedirs(folder)
        change_ownership(folder)

    dream_ID='dream_'+str(torch.randint(memory_dreams, (1,)).item())
    if not os.path.exists(os.path.join(dream_folder, dream_ID)):
        os.makedirs(os.path.join(dream_folder, dream_ID))
        change_ownership(os.path.join(folder, dream_ID))

    models=[]
    for subdir in os.listdir(folder):
        models.append(subdir)
    model_ID=random.sample(models,1)[0]

    dump = os.path.join(folder, model_ID, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #
    

    # -- log/checkpointing paths
    log_file = os.path.join(folder, model_ID, f'{tag}.csv')
    save_path = os.path.join(folder, model_ID, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, model_ID, f'{tag}-latest.pth.tar')
    dream_save_path = os.path.join(dream_folder, dream_ID, f'{tag}' + '-dream{dream}.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    print('log_file ', log_file)
    print('save_path ', save_path)
    print('latest_path ', latest_path)
    print('load_path ', load_path)

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))



    # -- init world model
    encoder, predictor = init_world_model(
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




    # -- init data-loader
    data = PTZImageDataset('./labels', './collected_imgs', transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    ipe = len(dataloader)



    for e in encoder.parameters():
        e.requires_grad = False

    for p in predictor.parameters():
        p.requires_grad = False




    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, _, _, _, _ = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor)




    def save_dreams(dream_id, dream_dict):
        torch.save(dream_dict, dream_save_path.format(dream=f'{dream_id}'))





    def get_random_position():
        if camera_brand==0:
            pan_pos = np.random.randint(0, 360)
            tilt_pos = np.random.randint(-90, 90)
            zoom_pos = np.random.randint(1, 2)
        elif camera_brand==1:
            pan_pos = np.random.randint(-180, 180)
            tilt_pos = np.random.randint(-180, 180)
            zoom_pos = np.random.randint(100, 200)

        return pan_pos, tilt_pos, zoom_pos






    def dream_step(inputs):
        images = inputs[0]
        possitions = inputs[1]
        number_of_dreams = inputs[2]

        state_sequences = []
        position_sequences = []
        reward_sequences = []
        action_sequences = []
        # Step 1. Forward image through encoder
        internal_state = get_internal_representation(images)
        # Step 2. Iterate forwarding internal representations through the predictor
        for step in range(dream_length):
            state_sequences.append(internal_state.unsqueeze(0))
            position_sequences.append(possitions.unsqueeze(0))

            next_possitions, next_actions = get_next_random_possitions(possitions, actions)
            next_internal_state, next_reward = forward_internal_representation(internal_state, possitions, next_possitions)

            internal_state = next_internal_state
            possitions = next_possitions
            reward = next_reward.squeeze(-1)
            reward = reward.mean(-1)
            reward_sequences.append(reward.unsqueeze(0))
            action_sequences.append(next_actions.unsqueeze(0))

        state_sequences.append(internal_state.unsqueeze(0))
        position_sequences.append(possitions.unsqueeze(0))

        B, D1, D2 = internal_state.shape
        aux = torch.Tensor(len(state_sequences), B, D1, D2).to(device)
        torch.cat(state_sequences, out=aux)
        state_sequences=aux

        B, D1 = possitions.shape
        aux = torch.Tensor(len(position_sequences), B, D1).to(device)
        torch.cat(position_sequences, out=aux)
        position_sequences=aux

        B = reward.shape[0]
        aux = torch.Tensor(len(reward_sequences), B).to(device)
        torch.cat(reward_sequences, out=aux)
        reward_sequences=aux

        B = next_actions.shape[0]
        aux = torch.Tensor(len(action_sequences), B).to(device)
        torch.cat(action_sequences, out=aux)
        action_sequences=aux

        for idx in range(B):
            dream_dict = {
                'state_sequence': state_sequences[:,idx],
                'possition_sequence': position_sequences[:,idx],
                'reward_sequence': reward_sequences[:,idx],
                'action_sequence': action_sequences[:,idx]
            }
            save_dreams(np.random.randint(number_of_dreams), dream_dict)

        change_ownership(dream_folder)

        return 0





    def get_internal_representation(images):
        z = encoder(images)
        return z

    def forward_internal_representation(internal_state, possition1, possition2):
        z, r = predictor(internal_state, possition1, possition2) 
        return z, r

    def choose_random_action(actions):
        action=np.random.randint(len(actions.keys()))
        return actions[action], action

    def get_next_random_possitions(possitions, actions_set):
        next_possitions = torch.zeros_like(possitions)
        next_actions = []
        for i, possition in enumerate(possitions):
            ptz_command, action = choose_random_action(actions_set)
            ptz_command = torch.tensor(ptz_command).to(device)
            next_actions.append(action)
            next_possitions[i] = possition + ptz_command

        next_actions=torch.tensor(next_actions).to(device)
        return next_possitions, next_actions



    def forward_target(images):
        h = target_encoder(images)
        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
        return h

    def forward_context(images, possition1, possition2):
        z = encoder(images)
        z, r = predictor(z, possition1, possition2) 
        return z, r


    def auxiliary_loss_fn(z, h):
        loss = F.smooth_l1_loss(z, h)
        return loss

    def loss_fn(z, r, h, g):
        loss1 = F.smooth_l1_loss(z, h)
        loss2 = F.smooth_l1_loss(r, g.repeat(r.shape))# * 1e-4
        loss = loss1 + loss2
        return loss



    def get_position_from_label(labels):
        possitions = []
        for label in labels:
            poss = label.split('_')[0].split(',')
            possitions.append([float(poss[0]), float(poss[1]), float(poss[2])])

        return torch.tensor(possitions)


    # Change allocentric position
    def change_allocentric_position(possition):
        pan, tilt, _ = get_random_position()
        possition[:,0] = possition[:,0] - pan
        possition[:,1] = possition[:,1] - tilt



    # -- DREAM LOOP
    for itr, (imgs, labls) in enumerate(dataloader):
        poss = get_position_from_label(labls)
        change_allocentric_position(poss)
        imgs = imgs.to(device, non_blocking=True)
        poss = poss.to(device, non_blocking=True)
        
        (dump), etime = gpu_timer(dream_step, arguments=[imgs, poss, number_of_dreams])
        if itr > number_of_dreams:
            break


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

    if mode=='train':
        return train(params, logger=logger)
    if mode=='world_model':
        return world_model(params, logger=logger)
    if mode=='dreamer':
        return dreamer(params, logger=logger)
    else:
        print(f"Unexpected mode {mode}")
        raise
