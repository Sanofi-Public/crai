import os
import sys

import time

import numpy as np
import scipy.spatial.distance
import torch
from torch.utils.data import DataLoader
import pymol
import pymol2

pymol.invocation.parse_args(['pymol', '-q'])  # optional, for quiet flag
pymol2.SingletonPyMOL().start()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from load_data.ABDataset import ABDataset
from learning.Unet import HalfUnetModel
from learning.SimpleUnet import SimpleHalfUnetModel
from utils.learning_utils import get_split_datasets, get_dataloaders, get_split_dataloaders, setup_learning
from utils.rotation import rotation_to_supervision
from utils.object_detection import nms
from utils.learning_utils import weighted_bce, weighted_dice_loss, weighted_focal_loss


def dists_to_hits(dists):
    """
    Take dists and turn it to a hit ratio
    0,1,2 => 1/3, 2/3
    0,0 => 2/2, 2/2
    :return:
    """
    dists = np.asarray(dists)
    hr_0 = np.sum(dists == 0) / len(dists)
    hr_1 = np.sum(dists <= 1) / len(dists)
    return hr_0, hr_1


def compute_metrics_ijks(actual_ijks, pred_ijks):
    dist_matrix = scipy.spatial.distance.cdist(pred_ijks, actual_ijks)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix)
    position_dists = dist_matrix[row_ind, col_ind]
    mean_dist = float(position_dists.mean())
    hr_0, hr_1 = dists_to_hits(position_dists)
    return mean_dist, hr_0, hr_1, position_dists, col_ind


def coords_loss(prediction, comp, classif_nano=True):
    """
    Object detection loss that accounts for finding the right voxel(s) and the right translation/rotation
       at this voxel.
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
    :param comp:
    :return:
    """

    metrics = {}
    pred_shape = prediction.shape[-3:]
    device = prediction.device
    prediction_np = prediction.clone().cpu().detach().numpy()

    # First let's find out the position of our antibodies in our prediction
    origin = comp.mrc.origin
    top = origin + comp.mrc.voxel_size * comp.mrc.data.shape
    bin_x = np.linspace(origin[0], top[0], num=pred_shape[0] + 1)
    bin_y = np.linspace(origin[1], top[1], num=pred_shape[1] + 1)
    bin_z = np.linspace(origin[2], top[2], num=pred_shape[2] + 1)

    # Compute the grid cell for each supervision
    # Some might fall out of the considered density : filter them out
    BCE_target = torch.zeros(size=pred_shape, device=device)
    filtered_transforms = []
    for rmsd, translation, rotation, nano in comp.transforms:
        position_i = np.digitize(translation[0], bin_x) - 1
        position_j = np.digitize(translation[1], bin_y) - 1
        position_k = np.digitize(translation[2], bin_z) - 1
        pos_tuple = (position_i, position_j, position_k)
        if all((0 <= pos_tuple[i] < pred_shape[i] for i in range(3))):
            BCE_target[position_i, position_j, position_k] = 1
            filtered_transforms.append((pos_tuple, translation, rotation, nano))
    # position_loss = weighted_bce(prediction[0, 0, ...], BCE_target, weights=[1, 1000])
    position_loss = weighted_focal_loss(prediction[0, 0, ...],
                                        BCE_target,
                                        weights=[1, 30],
                                        gamma=2)
    if len(filtered_transforms) == 0:
        return position_loss, None, None, None, None

    # Get the locations of the prediction
    actual_ijks = np.asarray([x[0] for x in filtered_transforms])
    prediction_np_loc = prediction_np[0, 0, ...]
    predicted_ijks_expanded = nms(pred_loc=prediction_np_loc, n_objects=max(5, len(filtered_transforms)))
    predicted_ijks_expanded = np.asarray(predicted_ijks_expanded)

    # As a metric, keep track of the bin distance using linear assignment. First compute it with 5 systems
    mean_dist_expanded, hr_0_expanded, hr_1_expanded, _, _ = compute_metrics_ijks(actual_ijks, predicted_ijks_expanded)
    metrics['mean_dist_5'] = mean_dist_expanded
    metrics['hr_0_5'] = hr_0_expanded
    metrics['hr_1_5'] = hr_1_expanded

    # Then again, with the right amount
    predicted_ijks = predicted_ijks_expanded[:len(filtered_transforms)]
    mean_dist, hr_0, hr_1, dists, mapping = compute_metrics_ijks(actual_ijks, predicted_ijks)
    metrics['mean_dist'] = mean_dist
    metrics['hr_0'] = hr_0
    metrics['hr_1'] = hr_1
    metrics['dists'] = dists

    actual_distances = []
    for index, (i, j, k) in enumerate(predicted_ijks):
        # Extract the predicted vector at this location
        ground_truth_translation = filtered_transforms[mapping[index]][1]
        predicted_offset = prediction_np[0, 1:4, i, j, k]
        predicted_position = predicted_offset + np.asarray([bin_x[i], bin_y[j], bin_z[k]])
        distance = np.linalg.norm(ground_truth_translation - predicted_position)
        actual_distances.append(distance)
    metrics['real_dists'] = actual_distances

    offset_losses, rz_losses, angle_losses, nano_losses = [], [], [], []
    for pos_tuple, translation, rotation, nano in filtered_transforms:
        # Extract the predicted vector at this location
        position_i, position_j, position_k = pos_tuple
        vector_pose = prediction[0, 1:, position_i, position_j, position_k]

        # Get the offset from the corner prediction loss
        offset_x = translation[0] - bin_x[position_i]
        offset_y = translation[1] - bin_y[position_j]
        offset_z = translation[2] - bin_z[position_k]
        gt_offset = torch.tensor([offset_x, offset_y, offset_z], device=device, dtype=torch.float)
        offset_loss = torch.nn.MSELoss()(vector_pose[:3], gt_offset)

        # Get the right pose. For that get the rotation supervision as a R3 vector and an angle.
        # We will penalise the R3 with its norm and it's dot product to ground truth
        rz, angle = rotation_to_supervision(rotation)
        rz = torch.tensor(rz, device=device, dtype=torch.float)
        predicted_rz = vector_pose[3:6]
        rz_norm = torch.norm(predicted_rz)
        rz_loss = 1 - torch.dot(predicted_rz / rz_norm, rz) + (rz_norm - 1) ** 2

        # Following AF2 and to avoid singularities, we frame the prediction of an angle as a regression task in the plane.
        # We turn our angle into a unit vector of R2, push predicted norm to and penalize dot product to ground truth
        vec_angle = [np.cos(angle), np.sin(angle)]
        vec_angle = torch.tensor(vec_angle, device=device, dtype=torch.float)
        predicted_angle = vector_pose[6:8]
        angle_norm = torch.norm(predicted_angle)
        angle_loss = 1 - torch.dot(predicted_angle / angle_norm, vec_angle) + (angle_norm - 1) ** 2

        if classif_nano:
            # Now we also include the nanobodies
            nano_loss = weighted_focal_loss(vector_pose[8], nano, weights=[1, 1726 / 426])
            # print(nano, vector_pose[8], nano_loss)
        else:
            nano_loss = torch.zeros(1)

        offset_losses.append(offset_loss)
        rz_losses.append(rz_loss)
        angle_losses.append(angle_loss)
        nano_losses.append(nano_loss)

    offset_loss = torch.mean(torch.stack(offset_losses))
    rz_loss = torch.mean(torch.stack(rz_losses))
    angle_loss = torch.mean(torch.stack(angle_losses))
    nano_loss = torch.mean(torch.stack(nano_losses))
    return position_loss, offset_loss, rz_loss, angle_loss, nano_loss, metrics


def dump_log(writer, epoch, to_log, prefix=''):
    for name, value in to_log.items():
        writer.add_scalar(f'{prefix}{name}', value, epoch)


def train(model, loader, optimizer, n_epochs=10, device='cpu',
          val_loader=None, val_loader_full=None, nano_loader=None,
          writer=None, accumulated_batch=1, save_path=''):
    best_mean_val_loss = 10000.
    last_model_path = f'{save_path}_last.pth'
    time_init = time.time()
    for epoch in range(n_epochs):
        for step, (name, comp) in enumerate(loader):
            if comp is None:
                continue
            input_tensor = torch.from_numpy(comp.input_tensor[None, ...]).to(device)
            prediction = model(input_tensor)
            position_loss, offset_loss, rz_loss, angle_loss, nano_loss, metrics = coords_loss(prediction, comp)
            position_dist = metrics['mean_dist']

            if offset_loss is None:
                loss = position_loss
            else:
                loss = position_loss + 0.2 * (0.3 * offset_loss + rz_loss + angle_loss + nano_loss)
            loss.backward()

            # Accumulated gradients
            if not step % accumulated_batch:
                optimizer.step()
                model.zero_grad()

            if not step % 100:
                step_total = len(loader) * epoch + step
                eluded_time = time.time() - time_init
                print(f"Epoch : {epoch} ; step : {step} ; loss : {loss.item():.5f} ; time : {eluded_time:.1f}")
                if offset_loss is not None:
                    to_log = {"loss": loss.item(),
                              "position_loss": position_loss.item(),
                              "offset_loss": offset_loss.item(),
                              "rz_loss": rz_loss.item(),
                              "angle_loss": angle_loss.item(),
                              "nano_loss": nano_loss.item(),
                              "position_distance": position_dist}
                    dump_log(writer, step_total, to_log, prefix='train_')

        if epoch == n_epochs - 1:
            model.cpu()
            torch.save(model.state_dict(), last_model_path)
            model.to(device)

        if val_loader is not None:
            print("validation")
            to_log = validate(model=model, device=device, loader=val_loader)
            val_loss = to_log["loss"]
            print(f'Validation loss ={val_loss}')
            dump_log(writer, epoch, to_log, prefix='val_')
            # Model checkpointing
            if val_loader_full is None and val_loss < best_mean_val_loss:
                best_mean_val_loss = val_loss
                best_model_path = f'{save_path}_{epoch}.pth'
                model.cpu()
                torch.save(model.state_dict(), best_model_path)
                model.to(device)

        if val_loader_full is not None:
            print("validation full")
            to_log = validate(model=model, device=device, loader=val_loader_full)
            val_loss = to_log["loss"]
            print(f'Validation loss ={val_loss}')
            dump_log(writer, epoch, to_log, prefix='full_val_')
            # Model checkpointing
            if val_loss < best_mean_val_loss:
                best_mean_val_loss = val_loss
                best_model_path = f'{save_path}_{epoch}.pth'
                model.cpu()
                torch.save(model.state_dict(), best_model_path)
                model.to(device)

        if nano_loader is not None:
            print("validation nano")
            to_log = validate(model=model, device=device, loader=nano_loader)
            val_loss = to_log["loss"]
            print(f'Validation loss ={val_loss}')
            dump_log(writer, epoch, to_log, prefix='nano_val_')
            if val_loader_full is None and val_loss < best_mean_val_loss:
                best_mean_val_loss = val_loss
                best_model_path = f'{save_path}_{epoch}.pth'
                model.cpu()
                torch.save(model.state_dict(), best_model_path)
                model.to(device)


def validate(model, device, loader):
    time_init = time.time()
    losses = list()
    all_real_dists = list()
    with torch.no_grad():
        for step, (name, comp) in enumerate(loader):
            if comp is None:
                continue
            input_tensor = torch.from_numpy(comp.input_tensor[None, ...]).to(device)
            prediction = model(input_tensor)
            position_loss, offset_loss, rz_loss, angle_loss, nano_loss, metrics = coords_loss(prediction, comp)
            position_dist = metrics['mean_dist']
            real_dists = metrics['real_dists']

            if offset_loss is not None:
                loss = position_loss + offset_loss + rz_loss + angle_loss + nano_loss
                losses.append([loss.item(),
                               position_loss.item(),
                               offset_loss.item(),
                               rz_loss.item(),
                               angle_loss.item(),
                               nano_loss.item(),
                               position_dist
                               ])
                all_real_dists.extend(real_dists)

            if not step % 100:
                print(f"step : {step} ; loss : {loss.item():.5f} ; time : {time.time() - time_init:.1f}")
    losses = np.array(losses)
    losses = np.mean(losses, axis=0)
    all_real_dists = np.asarray(all_real_dists)
    hr_6 = sum(all_real_dists <= 6) / len(all_real_dists)
    mean_real_dist = np.mean(all_real_dists)
    all_real_dists[all_real_dists >= 20] = 20
    mean_real_dist_capped = np.mean(all_real_dists)
    # print(hr_6)
    # print(mean_real_dist)
    # print(mean_real_dist_capped)

    to_log = {"loss": losses[0],
              "position_loss": losses[1],
              "offset_loss": losses[2],
              "rz_loss": losses[3],
              "angle_loss": losses[4],
              "nano_loss": losses[5],
              "position_distance": losses[6],
              "real_dist": mean_real_dist,
              "real_dist_capped": mean_real_dist_capped,
              "hr_6": hr_6,
              }
    return to_log


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", default='default')
    parser.add_argument("--train_full", action='store_false', default=True)
    parser.add_argument("--rotate", action='store_false', default=True)
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--agg_grads", type=int, default=4)
    parser.add_argument("--crop", type=int, default=3)
    args = parser.parse_args()

    writer, save_path, device = setup_learning(model_name=args.model_name,
                                               gpu_number=args.gpu)

    # Setup data
    use_fabs = True
    use_nano = True
    num_workers = 0
    # num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw
    # csv_train = "../data/csvs/chunked_train_reduced.csv"
    all_systems_train = []
    csv_train = []
    if use_fabs:
        all_systems_train.append("../data/csvs/filtered_train.csv")
        csv_train.append("../data/csvs/chunked_train.csv")
    if use_nano:
        all_systems_train.append("../data/nano_csvs/filtered_train.csv")
        csv_train.append("../data/nano_csvs/chunked_train.csv")
    train_ab_dataset = ABDataset(csv_to_read=csv_train, all_systems=all_systems_train,
                                 rotate=args.rotate, crop=args.crop, full=args.train_full)
    train_loader = DataLoader(dataset=train_ab_dataset, worker_init_fn=np.random.seed,
                              shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    # # Test loss
    # fake_out = torch.randn((1, 9, 23, 28, 19))
    # fake_out[0, 0, ...] = torch.sigmoid(fake_out[0, 0, ...])
    # coords_loss(fake_out, train_ab_dataset[0][1])
    # sys.exit()

    if use_fabs:
        csv_val = "../data/csvs/chunked_val.csv"
        # val_ab_dataset = ABDataset(csv_to_read=csv_val, rotate=False, crop=0, full=False)
        # val_loader = DataLoader(dataset=val_ab_dataset, collate_fn=lambda x: x[0], num_workers=num_workers)
        val_loader = None
        val_ab_dataset_full = ABDataset(csv_to_read=csv_val, rotate=False, crop=0, full=True)
        val_loader_full = DataLoader(dataset=val_ab_dataset_full, collate_fn=lambda x: x[0], num_workers=num_workers)
    else:
        val_loader = None
        val_loader_full = None
    if use_nano:
        csv_val_nano = "../data/nano_csvs/chunked_val.csv"
        all_system_val_nano = "../data/nano_csvs/filtered_val.csv"
        val_ab_dataset_nano_full = ABDataset(all_systems=all_system_val_nano, csv_to_read=csv_val_nano,
                                             rotate=False, crop=0, full=True)
        val_loader_nano_full = DataLoader(dataset=val_ab_dataset_nano_full, collate_fn=lambda x: x[0],
                                          num_workers=num_workers)
    else:
        val_loader_nano_full = None

    # Learning hyperparameters
    n_epochs = 1000
    accumulated_batch = args.agg_grads
    model = SimpleHalfUnetModel(in_channels=1,
                                classif_nano=use_nano and use_fabs,
                                out_channels=10,
                                model_depth=4,
                                num_convs=3,
                                max_decode=2,
                                num_feature_map=32)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train(model=model, loader=train_loader, optimizer=optimizer, n_epochs=n_epochs, device=device,
          val_loader=val_loader, val_loader_full=val_loader_full, nano_loader=val_loader_nano_full,
          writer=writer, accumulated_batch=accumulated_batch, save_path=save_path)
