import os
import sys

import time

import numpy as np
import torch
import pymol
import pymol2

pymol.invocation.parse_args(['pymol', '-q'])  # optional, for quiet flag
pymol2.SingletonPyMOL().start()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from load_data.ABDataset import ABDataset
from learning.Unet import HalfUnetModel
from utils.learning_utils import get_split_dataloaders, rotation_to_supervision
from learning.train_functions import train, setup_learning


def coords_loss(prediction, complex):
    """
    Object detection loss that accounts for finding the right voxel and the right rotation at this voxel.
    It is a sum of several components :
    position_loss = BCE(predicted_objectness, gt_presence)
        find the right voxel
    offset_loss = MSE(predicted_offset, gt_offset)
        find the right offset from the corner of this voxel
    rz_loss = 1 - <predicted_rz, gt_rz> - MSE(predicted_rz, 1)
        find the right rotation of the main axis
    angle_loss = 1 - <predicted_angle, point(gt_angle)> - MSE(predicted_angle, 1)
        find the right rotation around the main axis, with AF2 trick to predict angles
    :param prediction:
    :param complex:
    :return:
    """
    pred_shape = prediction.shape[-3:]
    device = prediction.device

    # First let's find out the position of the antibody in our prediction
    origin = complex.mrc.origin
    top = origin + complex.mrc.voxel_size * complex.mrc.data.shape
    bin_x = np.linspace(origin[0], top[0], num=pred_shape[0] + 1)
    bin_y = np.linspace(origin[1], top[1], num=pred_shape[1] + 1)
    bin_z = np.linspace(origin[2], top[2], num=pred_shape[2] + 1)
    position_x = np.digitize(complex.translation[0], bin_x) - 1
    position_y = np.digitize(complex.translation[1], bin_y) - 1
    position_z = np.digitize(complex.translation[2], bin_z) - 1

    # Now let's add finding this spot as a loss term
    BCE_target = torch.zeros(size=pred_shape, device=device)
    BCE_target[position_x, position_y, position_z] = 1
    position_loss = torch.nn.BCELoss()(prediction[0, 0, ...], BCE_target)

    # Extract the predicted vector at this location
    vector_pose = prediction[0, 1:, position_x, position_y, position_z]

    # Get the offset from the corner prediction loss
    offset_x = complex.translation[0] - bin_x[position_x]
    offset_y = complex.translation[1] - bin_y[position_y]
    offset_z = complex.translation[2] - bin_z[position_z]
    gt_offset = torch.Tensor([offset_x, offset_y, offset_z], device=device)
    offset_loss = torch.nn.MSELoss()(vector_pose[:3], gt_offset)

    # Get the right pose. For that get the rotation supervision as a R3 vector and an angle.
    # We will penalise the R3 with its norm and it's dot product to ground truth
    rz, angle = rotation_to_supervision(complex.rotation)
    rz = torch.Tensor(rz, device=device)
    predicted_rz = vector_pose[3:6]
    rz_norm = torch.norm(predicted_rz)
    rz_loss = 1 - torch.dot(predicted_rz / rz_norm, rz) + (rz_norm - 1) ** 2

    # Following AF2 and to avoid singularities, we frame the prediction of an angle as a regression task in the plane.
    # We turn our angle into a unit vector of R2, push predicted norm to and penalize dot product to ground truth
    vec_angle = [np.cos(angle), np.sin(angle)]
    vec_angle = torch.Tensor(vec_angle, device=device)
    predicted_angle = vector_pose[6:]
    angle_norm = torch.norm(predicted_angle)
    angle_loss = 1 - torch.dot(predicted_angle / rz_norm, vec_angle) + (angle_norm - 1) ** 2

    loss = position_loss + offset_loss + rz_loss + angle_loss
    return loss


def loop_fn(model, device, complex):
    input_tensor = torch.from_numpy(complex.input_tensor[None, ...]).to(device)
    prediction = model(input_tensor)
    loss = coords_loss(prediction, complex)
    return loss


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", default='default')
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    writer, best_model_path, last_model_path, device = setup_learning(model_name=args.model_name,
                                                                      gpu_number=args.gpu)

    # Setup data
    rotate = True
    num_workers = 0
    # num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw
    data_root = "../data/pdb_em"
    csv_to_read = "../data/reduced_final.csv"
    # csv_to_read = "data/final.csv"
    ab_dataset = ABDataset(data_root=data_root,
                           csv_to_read=csv_to_read,
                           rotate=rotate,
                           return_grid=False)

    fake_out = torch.randn((1, 9, 23, 28, 19))
    fake_out[0, 0, ...] = torch.sigmoid(fake_out[0, 0, ...])
    coords_loss(fake_out, ab_dataset[0][1])
    train_loader, val_loader, _ = get_split_dataloaders(dataset=ab_dataset,
                                                        shuffle=True,
                                                        collate_fn=lambda x: x[0],
                                                        num_workers=num_workers)

    # Learning hyperparameters
    n_epochs = 700
    accumulated_batch = 5
    model = HalfUnetModel(out_channels_decoder=128,
                          num_feature_map=24, )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Train
    train(model=model, device=device, loop_fn=loop_fn, loader=train_loader,
          optimizer=optimizer, writer=writer, n_epochs=n_epochs, val_loader=val_loader,
          accumulated_batch=accumulated_batch, save_path=best_model_path)
    model.cpu()
    torch.save(model.state_dict(), last_model_path)
