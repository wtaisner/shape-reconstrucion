import gc
import os
import random
import time

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from datetime import datetime as dt

from src.Pix2Vox.models.decoder import Decoder
from src.Pix2Vox.models.encoder import Encoder
from src.Pix2Vox.models.merger import Merger
from src.Pix2Vox.models.refiner import Refiner
from src.Pix2Vox.shapenet_dataset import ShapeNetDataset
from src.Pix2Vox.utils import data_transforms, network_utils
from src.utils import read_config, seed_worker

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(42)
    cfg = read_config("../config/pix2vox.yaml")
    print(cfg)
    wandb.init(
        # set the wandb project where this run will be logged
        entity='ap-wt',
        project="shape-reconstruction",
        # track hyperparameters and run metadata
        config=cfg
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    cpu = torch.device("cpu")
    img_size = cfg['dataset']['img_height'], cfg['dataset']['img_width']
    crop_size = cfg['dataset']['crop_height'], cfg['dataset']['crop_width']

    if cfg['train_params']['augment']:
        train_transforms = data_transforms.Compose([
            data_transforms.RandomCrop(img_size, crop_size),
            data_transforms.RandomBackground(cfg['train_params']['transform']['random_bg_color_range']),
            data_transforms.ColorJitter(cfg['train_params']['transform']['brightness'],
                                        cfg['train_params']['transform']['contrast'],
                                        cfg['train_params']['transform']['saturation']),
            data_transforms.RandomNoise(cfg['train_params']['transform']['noise_std']),
            data_transforms.Normalize(mean=cfg['dataset']['mean'], std=cfg['dataset']['std']),
            data_transforms.RandomFlip(),
            data_transforms.ToTensor(),
        ])
        test_transforms = data_transforms.Compose([
            data_transforms.CenterCrop(img_size, crop_size),
            data_transforms.RandomBackground(cfg['test_params']['transform']['random_bg_color_range']),
            data_transforms.Normalize(mean=cfg['dataset']['mean'], std=cfg['dataset']['std']),
            data_transforms.ToTensor(),
        ])
    else:
        train_transforms = data_transforms.Compose([
            data_transforms.CenterCrop(img_size, crop_size),
            data_transforms.Normalize(mean=cfg['dataset']['mean'], std=cfg['dataset']['std']),
            data_transforms.ToTensor(),
        ])
        test_transforms = data_transforms.Compose([
            data_transforms.CenterCrop(img_size, crop_size),
            data_transforms.Normalize(mean=cfg['dataset']['mean'], std=cfg['dataset']['std']),
            data_transforms.ToTensor(),
        ])

    train_dataset = ShapeNetDataset(cfg['dataset']['train_data_file'], cfg['dataset']['img_path'],
                                    cfg['dataset']['models_path'], train_transforms)
    test_dataset = ShapeNetDataset(cfg['dataset']['eval_data_file'], cfg['dataset']['img_path'],
                                   cfg['dataset']['models_path'], test_transforms)

    g_train, g_test = torch.Generator(), torch.Generator()
    g_train.manual_seed(42)
    g_test.manual_seed(42)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["train_params"]["batch_size"],
        num_workers=cfg["train_params"]["num_workers"],
        worker_init_fn=seed_worker,
        generator=g_train,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg["train_params"]["batch_size"],
        num_workers=cfg["train_params"]["num_workers"],
        worker_init_fn=seed_worker,
        generator=g_test,
        shuffle=False
    )

    encoder = Encoder(cfg['network'])
    decoder = Decoder(cfg['network'])
    refiner = Refiner(cfg['network'])
    merger = Merger(cfg['network'])

    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), network_utils.count_parameters(refiner)))
    print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), network_utils.count_parameters(merger)))

    encoder.apply(network_utils.init_weights)
    decoder.apply(network_utils.init_weights)
    refiner.apply(network_utils.init_weights)
    merger.apply(network_utils.init_weights)

    encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                      lr=cfg['train_params']['encoder_lr'],
                                      betas=cfg['train_params']['betas'])
    decoder_solver = torch.optim.Adam(decoder.parameters(),
                                      lr=cfg['train_params']['decoder_lr'],
                                      betas=cfg['train_params']['betas'])
    refiner_solver = torch.optim.Adam(refiner.parameters(),
                                      lr=cfg['train_params']['refiner_lr'],
                                      betas=cfg['train_params']['betas'])
    merger_solver = torch.optim.Adam(merger.parameters(), lr=cfg['train_params']['merger_lr'],
                                     betas=cfg['train_params']['betas'])

    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=cfg['train_params']['encoder_lr_milestones'],
                                                                gamma=cfg['train_params']['gamma'])
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                milestones=cfg['train_params']['decoder_lr_milestones'],
                                                                gamma=cfg['train_params']['gamma'])
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver,
                                                                milestones=cfg['train_params']['refiner_lr_milestones'],
                                                                gamma=cfg['train_params']['gamma'])
    merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver,
                                                               milestones=cfg['train_params']['merger_lr_milestones'],
                                                               gamma=cfg['train_params']['gamma'])
    encoder.to(device)
    decoder.to(device)
    refiner.to(device)
    merger.to(device)

    bce_loss = torch.nn.BCELoss()

    init_epoch = 0
    best_iou = -1
    best_epoch = -1

    if len(cfg['train_params']['weights']) != 0 and cfg['train_params']['resume']:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg['train_params']['weights']))
        checkpoint = torch.load(cfg['train_params']['weights'])
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if cfg['network']['use_refiner']:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg['network']['use_merger']:
            merger.load_state_dict(checkpoint['merger_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
              (dt.now(), init_epoch, best_iou, best_epoch))

    output_dir = os.path.join(cfg['train_params']['output_dir'], '%s', dt.now().isoformat())
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'

    print('training started')
    for epoch in range(init_epoch, cfg['train_params']['num_epochs']):
        epoch_start_time = time.time()

        batch_time = network_utils.AverageMeter()
        data_time = network_utils.AverageMeter()
        encoder_losses = network_utils.AverageMeter()
        refiner_losses = network_utils.AverageMeter()

        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()

        batch_end_time = time.time()

        n_batches = len(train_dataloader)
        for batch_idx, (taxonomy_names, sample_names, imgs, gt_volumes) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_end_time)

            imgs = imgs.to(device)
            gt_volumes = gt_volumes.to(device)

            img_features = encoder(imgs)
            raw_features, generated_volumes = decoder(img_features)

            if cfg['network']['use_merger'] and epoch >= cfg['train_params']['epoch_start_use_merger']:
                generated_volumes = merger(raw_features, generated_volumes)
            else:
                generated_volumes = torch.mean(generated_volumes, dim=1)
            encoder_loss = bce_loss(generated_volumes, gt_volumes) * 10

            if cfg['network']['use_refiner'] and epoch >= cfg['train_params']['epoch_start_use_refiner']:
                generated_volumes = refiner(generated_volumes)
                refiner_loss = bce_loss(generated_volumes, gt_volumes) * 10
            else:
                refiner_loss = encoder_loss

            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            refiner.zero_grad()
            merger.zero_grad()

            if cfg['network']['use_refiner'] and epoch >= cfg['train_params']['epoch_start_use_refiner']:
                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()
            else:
                encoder_loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            merger_solver.step()

            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            batch_time.update(time.time() - batch_end_time)
            batch_end_time = time.time()

        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()

        epoch_end_time = time.time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
              (
                  dt.now(), epoch, cfg['train_params']['num_epochs'], epoch_end_time - epoch_start_time,
                  encoder_losses.avg,
                  refiner_losses.avg))

        wandb.log({'train/EncoderDecoderLoss': encoder_losses.avg, 'train/RefinerLoss': refiner_losses.avg}, step=epoch)

        encoder.eval()
        decoder.eval()
        refiner.eval()
        merger.eval()

        test_ious = {}
        test_encoder_losses = network_utils.AverageMeter()
        test_refiner_losses = network_utils.AverageMeter()
        for batch_idx, (taxonomy_names, sample_names, imgs, gt_volumes) in enumerate(
                test_dataloader):

            with torch.no_grad():
                imgs = imgs.to(device)
                gt_volumes = gt_volumes.to(device)

                # Test the encoder, decoder, refiner and merger
                image_features = encoder(imgs)
                raw_features, generated_volume = decoder(image_features)

                if cfg['network']['use_merger'] and epoch >= cfg['train_params']['epoch_start_use_merger']:
                    generated_volume = merger(raw_features, generated_volume)
                else:
                    generated_volume = torch.mean(generated_volume, dim=1)
                encoder_loss = bce_loss(generated_volume, gt_volumes) * 10

                if cfg['network']['use_refiner'] and epoch >= cfg['train_params']['epoch_start_use_refiner']:
                    generated_volume = refiner(generated_volume)
                    refiner_loss = bce_loss(generated_volume, gt_volumes) * 10
                else:
                    refiner_loss = encoder_loss

                test_encoder_losses.update(encoder_loss.item())
                test_refiner_losses.update(refiner_loss.item())

                for taxonomy, sample, generated, gt in zip(taxonomy_names, sample_names, generated_volume, gt_volumes):
                    test_iou = []
                    for th in cfg['test_params']['voxel_thr']:
                        _volume = torch.ge(generated_volume, th).float()
                        intersection = torch.sum(_volume.mul(gt_volumes)).float()
                        union = torch.sum(torch.ge(_volume.add(gt_volumes), 1)).float()
                        test_iou.append((intersection / union).tolist())
                    if taxonomy not in test_ious:
                        test_ious[taxonomy] = [test_iou]
                    else:
                        test_ious[taxonomy].append(test_iou)
        mean_ious_taxonomy = {tax: np.mean(np.array(test_ious[tax]), axis=0) for tax in test_ious}
        test_iou_values = np.array([a for b in list(test_ious.values()) for a in b])
        mean_ious_all = np.mean(test_iou_values, axis=0)
        max_iou = np.max(mean_ious_all)
        print('=========================')
        print(f'[INFO] {dt.now()} Test [{epoch}/{cfg["train_params"]["num_epochs"]}] Partial results')
        for key, value in mean_ious_taxonomy.items():
            print(f'Taxonomy {key}, mean IOU {value}')
        print(f"Thresholds:{cfg['test_params']['voxel_thr']} Mean IOU: {mean_ious_all}")
        print('IOU = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
              (max_iou, test_encoder_losses.avg, test_refiner_losses.avg))
        print('=========================')
        metrics = dict((f'test/IOU_thr_{th}', v) for th, v in zip(cfg["test_params"]["voxel_thr"], mean_ious_all))
        metrics['test/EncoderDecoderLoss'] = test_encoder_losses.avg
        metrics['test/RefinerLoss'] = test_refiner_losses.avg
        metrics['test/IOU'] = max_iou
        wandb.log(metrics, step=epoch)

        if epoch % cfg['train_params']['save_every_n_epochs'] == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % epoch),
                                           epoch, encoder, encoder_solver, decoder, decoder_solver,
                                           refiner, refiner_solver, merger, merger_solver, best_iou,
                                           best_epoch)
        if max_iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            best_iou = max_iou
            best_epoch = epoch
            network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'best-ckpt.pth'), epoch,
                                           encoder, encoder_solver, decoder, decoder_solver, refiner,
                                           refiner_solver, merger, merger_solver, best_iou, best_epoch)

        if device == torch.device("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    wandb.finish()
