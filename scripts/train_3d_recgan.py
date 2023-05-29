import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.RecGAN import RecGAN
from src.RecGAN.shapenet_dataset import ShapeNetDataset
from src.utils import read_config, calculate_gradient_penalty, seed_worker

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

    model = RecGAN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['step_size'],
        gamma=config['gamma'])

    for e in tqdm(range(config["epochs"]), total=config['epochs'], desc="Epochs ",position=0, leave=False):
        model.train()
        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training ", leave=False, position=1):
            batch = batch.to(device)

            # train generator
            generator_predictions = model(batch)
            generator_loss = torch.mean(
                -torch.mean(config['ae_weight'] * batch * torch.log(generator_predictions[0] + 1e-8), dim=1) -
                torch.mean((1 - config['ae_weight']) * (1 - batch) * torch.log(generator_predictions[0] + 1e-8), dim=1)
            )
            # train discriminator
            discriminator_predictions = model(generator_predictions[0], batch, train_D=True)
            gradient_penalty = calculate_gradient_penalty(model, batch, generator_predictions[0], device)
            discriminator_loss = torch.mean(generator_predictions[0]) - torch.mean(batch) + gradient_penalty
            (generator_loss + discriminator_loss).backward()

            optimizer.step()
            optimizer.zero_grad()
            if device == torch.device("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        train_dataset.plot_from_voxels(generator_predictions[0].squeeze(0).squeeze(0).detach().cpu().numpy())
        # model.eval()
        # with torch.inference_mode():
        #     ground_truth, predictions = None, None
        #     for idx, batch in enumerate(test_dataloader):
        #         torch.cuda.empty_cache()
        #
        #         tmp_predictions = model(batch)
        #
        #         if ground_truth is None:
        #             ground_truth = TODO
        #             predictions = tmp_predictions
        #         else:
        #             try:
        #                 ground_truth = torch.cat([ground_truth, TODO], 0)
        #                 predictions = torch.cat([predictions, tmp_predictions], 0)
        #             except:
        #                 pass

        scheduler.step()
