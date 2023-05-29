from datetime import datetime as dt

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.Pix2Vox.models.decoder import Decoder
from src.Pix2Vox.models.encoder import Encoder
from src.Pix2Vox.models.merger import Merger
from src.Pix2Vox.models.refiner import Refiner
from src.Pix2Vox.shapenet_dataset import ShapeNetDataset
from src.Pix2Vox.utils import data_transforms, network_utils
from src.utils import read_config, seed_worker

if __name__ == '__main__':
    cfg = read_config("../config/pix2vox.yaml")
    wandb.init(
        # set the wandb project where this run will be logged
        entity='ap-wt',
        project="shape-reconstruction",
        # track hyperparameters and run metadata
        config=cfg,
        name="test-run-29"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    img_size = cfg['dataset']['img_height'], cfg['dataset']['img_width']
    crop_size = cfg['dataset']['crop_height'], cfg['dataset']['crop_width']

    test_transforms = data_transforms.Compose([
        data_transforms.CenterCrop(img_size, crop_size),
        data_transforms.Normalize(mean=cfg['dataset']['mean'], std=cfg['dataset']['std']),
        data_transforms.ToTensor(),
    ])

    test_dataset = ShapeNetDataset(cfg['dataset']['test_data_file'], cfg['dataset']['img_path'],
                                   cfg['dataset']['models_path'], test_transforms)

    g_test = torch.Generator()
    g_test.manual_seed(42)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg['train_params']['batch_size'],
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

    encoder.to(device)
    decoder.to(device)
    refiner.to(device)
    merger.to(device)

    bce_loss = torch.nn.BCELoss()

    print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg['train_params']['weights']))
    checkpoint = torch.load(cfg["weights_path"])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if cfg['network']['use_refiner']:
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
    if cfg['network']['use_merger']:
        merger.load_state_dict(checkpoint['merger_state_dict'])

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

            if cfg['network']['use_merger']:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)
            encoder_loss = bce_loss(generated_volume, gt_volumes) * 10

            if cfg['network']['use_refiner']:
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
    print(f'[INFO] {dt.now()} Test')
    for key, value in mean_ious_taxonomy.items():
        print(f'Taxonomy {key}, mean IOU {value}')
        wandb.log({f"Taxonomy {key} mean IOU": str(value)})
    print(f"Thresholds:{cfg['test_params']['voxel_thr']} Mean IOU: {mean_ious_all}")
    print(f'Max IOU for: {cfg["test_params"]["voxel_thr"][np.argmax(mean_ious_all)]}, Max IOU: {np.max(mean_ious_all)}')
    print(f'Encoder loss: {test_encoder_losses.avg}')
    print(f'Refiner loss: {test_refiner_losses.avg}')

    wandb.log({"EncoderDecoderLoss": test_encoder_losses.avg})
    wandb.log({"RefinerLoss": test_refiner_losses.avg})
