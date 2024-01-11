import os

import numpy as np
import string

import torch
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from geomloss import sinkhorn_divergence

ALLOWED_CHARS = set(string.ascii_letters + string.digits + '._-/')


def weighted_ce_loss(output, target, weight=None):
    """
    :param output:
    :param target:
    :return:
    """
    grid_loss_values = target * torch.log(output + 1e-5)
    if weight is None:
        return -torch.mean(grid_loss_values)
    expanded = weight.expand_as(target)
    return -torch.mean(expanded * grid_loss_values)


def weighted_dice_loss(output, target, weight=None):
    """
    Return weighted Dice loss for the HD branch
    :param output:
    :param target:
    :return:
    """

    grid_loss_values = 1 - (2 * target * output + 0.01) / (target + output + 0.01)
    if weight is None:
        return torch.mean(grid_loss_values)
    expanded = weight.expand_as(target)
    return torch.mean(expanded * grid_loss_values)


def weighted_bce(output, target, weights=None):
    output = torch.clamp(output, min=1e-5, max=1 - 1e-5)
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def weighted_focal_loss(output, target, weights=None, gamma=2):
    """

    :param output:
    :param target:
    :param weights: an iterable with weight_negatives, weight_positives
    :param gamma:
    :return:
    """
    output = torch.clamp(output, min=1e-5, max=1 - 1e-5)
    if weights is not None:
        assert len(weights) == 2

        loss1 = weights[1] * (target * (1 - output) ** gamma * torch.log(output))
        loss0 = weights[0] * ((1 - target) * output ** gamma * torch.log(1 - output))
        loss = loss0 + loss1
    else:
        loss = target * (1 - output) ** gamma * torch.log(output) + \
               (1 - target) * output ** gamma * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def ot_loss(output, target):
    output = output / torch.sum(output)
    target = target / torch.sum(output)
    grid_shape = output.shape
    # closest superior power of two
    n = 2 ** (1 + int(np.log2(max(grid_shape) - 0.5)))
    pad = torch.nn.ConstantPad3d((0, n - grid_shape[2], 0, n - grid_shape[1], 0, n - grid_shape[0]), 0)
    output = pad(output)
    target = pad(target)
    ot_loss = sinkhorn_divergence(output[None, None, ...], target[None, None, ...], scaling=0.9)
    return ot_loss


def get_split_datasets(dataset,
                       split_train=0.7,
                       split_valid=0.85,
                       ):
    n = len(dataset)
    np.random.seed(0)
    torch.manual_seed(0)

    train_index, valid_index = int(split_train * n), int(split_valid * n)
    indices = list(range(n))

    train_indices = indices[:train_index]
    valid_indices = indices[train_index:valid_index]
    test_indices = indices[valid_index:]

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)
    return train_set, valid_set, test_set


def get_dataloaders(datasets, **kwargs):
    loaders = []
    for dataset in datasets:
        loaders.append(DataLoader(dataset=dataset, worker_init_fn=np.random.seed, **kwargs))
    return loaders


def get_split_dataloaders(dataset,
                          split_train=0.7,
                          split_valid=0.85,
                          **kwargs):
    n = len(dataset)
    np.random.seed(0)
    torch.manual_seed(0)

    train_index, valid_index = int(split_train * n), int(split_valid * n)
    indices = list(range(n))

    train_indices = indices[:train_index]
    valid_indices = indices[train_index:valid_index]
    test_indices = indices[valid_index:]

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(dataset=train_set, worker_init_fn=np.random.seed, **kwargs)
    valid_loader = DataLoader(dataset=valid_set, **kwargs)
    test_loader = DataLoader(dataset=test_set, **kwargs)
    return train_loader, valid_loader, test_loader


def setup_learning(model_name, gpu_number):
    # Check each character in the input to avoid injection
    if all(char in ALLOWED_CHARS for char in model_name):
        sanitized_input = model_name
    else:
        raise ValueError("Input contains invalid characters")

    os.makedirs("../saved_models", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"../logs/{sanitized_input}")
    save_path = os.path.join("../saved_models", sanitized_input)
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'
    return writer, save_path, device
