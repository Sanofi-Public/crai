import random

import numpy as np
from scipy.spatial.transform import Rotation


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


def vector_to_rotation(rz, use_uy=False):
    """
    Compute the minimum rotation M turning uz into rz.
    """
    if use_uy:
        uz = np.array([0, 1, 0])
    else:
        uz = np.array([0, 0, 1])
    cross = np.cross(uz, rz)
    cross = cross / np.linalg.norm(cross)
    angle = np.arccos(np.dot(uz, rz))
    uz_to_p = Rotation.from_rotvec(angle * cross)
    possible_p = uz_to_p.apply(uz)
    if not np.dot(possible_p, rz) > 0.999:
        uz_to_p = Rotation.from_rotvec(-angle * cross)
    # assert p_prime == p
    # p_prime = uz_to_p.apply(uz)
    return uz_to_p


def rotation_to_supervision(rotation, use_uy=False):
    # Let's decompose our rotation into the minimum affecting uz : M = uz_to_p and an angle t rotation around p.
    # rot(p,t) = (uz_to_p)^-1 * rotation
    matrix = rotation.as_matrix()
    rz = matrix[:, 1 if use_uy else 2]
    uz_to_p = vector_to_rotation(rz, use_uy=use_uy)

    # Now we can decompose the rotation p -> p' and a rotation around p of angle t.
    rot_pt = uz_to_p.inv() * rotation
    pt = rot_pt.as_rotvec()
    return rz, pt[1 if use_uy else 2]
