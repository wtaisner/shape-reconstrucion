import random
from typing import Dict

import numpy as np
import torch
import yaml


def read_config(path: str) -> Dict:
    with open(path) as fp:
        cfg = yaml.safe_load(fp)
    return cfg


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_gradient_penalty(
        model: torch.nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        device: torch.device
) -> torch.Tensor:
    """
    Source: https://github.com/Lornatang/WassersteinGAN_GP-PyTorch/blob/master/wgangp_pytorch/utils.py#L39
    Calculates the gradient penalty loss for WGAN GP
    """
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates, _ = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty
