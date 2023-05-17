import numpy as np
import random
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Subset, DataLoader


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


class Rotor:
    def __init__(self, rotate_around_z=None, rotate_in_plane=None):
        """
        There are 8 admissible SO(3) rotations, first a complete flip around an axis then a p4 in the plane
        The Cayley diagram can be seen here : https://arxiv.org/abs/1804.04656
        :return: The 2 parameters for a rotation, along with a rotation representation of this operation
        """
        if rotate_around_z is None or rotate_in_plane is None:
            rotate_around_z = random.randint(0, 1)
            rotate_in_plane = random.randint(0, 3)
        rz = Rotation.from_rotvec(rotate_around_z * np.pi * np.array([0, 0, 1]))
        r2 = Rotation.from_rotvec(rotate_in_plane * np.pi / 2 * np.array([1, 0, 0]))
        r_tot = r2 * rz
        self.rotate_around_z = rotate_around_z
        self.rotate_in_plane = rotate_in_plane
        self.r_tot = r_tot

    def rotate_tensor(self, tensor):
        """
        :param tensor: The tensor to rotate
        :return:
        """

        rot1 = np.rot90(tensor, k=2 * self.rotate_around_z, axes=(-3, -2))
        rot2 = np.rot90(rot1, k=self.rotate_in_plane, axes=(-2, -1))
        return np.ascontiguousarray(rot2)

    def rotate_tensors(self, tensors):
        """
        Simultaneously rotate tensors
        :param tensors: The tensors to rotate, as a list
        :return:
        """
        assert isinstance(tensors, list)
        tensors = [self.rotate_tensor(tensor) for tensor in tensors]
        return tensors

    def rotate_with_origin(self, tensor, origin, voxel_size: float):
        """
        Useful for MRC data : keep track of the bottom left corner of an array in xyz space
        We assume regular voxel size
        :param tensor:
        :param origin:
        :param voxel_size: for now, this is only supported for regular voxel sizes, otherwise we need to keep track of
        the actual rotations
        :return:
        """
        top_corner = origin + voxel_size * tensor.shape
        rot_origin = self.r_tot.apply(origin)
        rot_top = self.r_tot.apply(top_corner)
        new_origin = np.min((rot_origin, rot_top), axis=0)
        new_tensor = self.rotate_tensor(tensor)
        return new_tensor, new_origin

    def rotate_around_origin(self, tensor, origin, voxel_size: float):
        """
        Useful for MRC data : keep track of the bottom left corner of an array in xyz space
        We assume rotation around the origin and regular voxel size
        :param tensor:
        :param origin:
        :param voxel_size: for now, this is only supported for regular voxel sizes, otherwise we need to keep track of
        the actual rotations
        :return:
        """
        top_corner = voxel_size * tensor.shape
        rot_top = self.r_tot.apply(top_corner) + origin
        new_origin = np.min((origin, rot_top), axis=0)
        new_tensor = self.rotate_tensor(tensor)
        return new_tensor, new_origin


def vector_to_angle(p):
    """
    Compute the minimum rotation turning uz into p
    """
    uz = np.array([0, 0, 1])
    cross = np.cross(uz, p)
    angle = np.arccos(np.dot(uz, p))
    uz_to_p = Rotation.from_rotvec(angle * cross)
    possible_p = uz_to_p.apply(uz)
    if np.dot(possible_p, p) < 0.999:
        angle = -angle
    return angle


def rotation_to_supervision(rotation):
    matrix = rotation.as_matrix()
    rz = matrix[:, 2]
    angle = vector_to_angle(rz)
    return rz, angle
