import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.RecGAN import RecGAN
from src.shapenet_dataset import ShapeNetDataset
from src.utils import read_config, calculate_gradient_penalty


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    torch.manual_seed(23)
    config = read_config("../config/3d_recgan.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")

    train_dataset = ShapeNetDataset(config["train"], vox_res=config["vox_res"])
    test_dataset = ShapeNetDataset(config["test"], vox_res=config["vox_res"])

    g_train, g_test = torch.Generator(), torch.Generator()
    g_train.manual_seed(42)
    g_test.manual_seed(42)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["no_workers"],
        worker_init_fn=seed_worker,
        generator=g_train,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["no_workers"],
        worker_init_fn=seed_worker,
        generator=g_test,
        shuffle=True
    )

    model = RecGAN().to(device)  # TODO Future: add channels argument
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['step_size'],
        gamma=config['gamma'])

    for e in tqdm(range(config["epochs"]), total=config['epochs']):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            if idx % 10 == 0:
                print(idx)
            batch = batch.to(device)
            # TODO: determine how to properly train both discriminator and generator

            # train generator
            generator_predictions = model(batch)
            # loss_generator TODO: can be substituted by BCELoss with weights, if you know how to set them XD
            generator_loss = torch.mean(
                -torch.mean(config['ae_weight'] * batch * torch.log(generator_predictions[0] + 1e-8), dim=1) -
                torch.mean((1 - config['ae_weight']) * (1 - batch) * torch.log(generator_predictions[0] + 1e-8), dim=1)
            )
            # train discriminator
            discriminator_predictions = model(generator_predictions[0], batch, train_D=True)
            gradient_penalty = calculate_gradient_penalty(model, batch, generator_predictions[0], device)
            # TODO: define discriminator loss function
            discriminator_loss = torch.mean(generator_predictions[0]) - torch.mean(batch) + gradient_penalty
            (generator_loss + discriminator_loss).backward()  # TODO: optionally, backward each loss separately?

            optimizer.step()
            optimizer.zero_grad()
            if device == torch.device("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        model.eval()
        with torch.inference_mode():
            ground_truth, predictions = None, None
            for idx, batch in enumerate(test_dataloader):
                torch.cuda.empty_cache()

                tmp_predictions = model(batch)

                if ground_truth is None:
                    ground_truth = TODO
                    predictions = tmp_predictions
                else:
                    try:
                        ground_truth = torch.cat([ground_truth, TODO], 0)
                        predictions = torch.cat([predictions, tmp_predictions], 0)
                    except:
                        pass

        scheduler.step()
