import numpy as np
import scipy.optimize
import scipy.spatial
import torch

from utils.learning_utils import weighted_focal_loss, ot_loss
from utils.object_detection import nms
from utils.rotation import rotation_to_supervision


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
    return mean_dist, hr_0, hr_1, position_dists, row_ind, col_ind


def coords_loss(prediction, comp, classif_nano=True, ot_weight=1., use_threshold=False, use_pd=False):
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
                                        gamma=4)
    if ot_weight != 0:
        ot_loss_value = ot_weight * ot_loss(prediction[0, 0, ...], BCE_target)[0]
        # print(ot_loss_value)
        position_loss += ot_loss_value
    if len(filtered_transforms) == 0:
        return position_loss, None, None, None, None, None

    # Get the locations of the prediction
    actual_ijks = np.asarray([x[0] for x in filtered_transforms])
    prediction_np_loc = prediction_np[0, 0, ...]
    if use_threshold:
        predicted_ijks_expanded = nms(pred_loc=prediction_np_loc, use_pd=use_pd)
    else:
        predicted_ijks_expanded = nms(pred_loc=prediction_np_loc,
                                      n_objects=max(5, len(filtered_transforms)),
                                      use_pd=use_pd)
        predicted_ijks_expanded = np.asarray(predicted_ijks_expanded)

    # This can happen with thresholds
    if len(predicted_ijks_expanded) == 0:
        metrics['mean_dist'] = 20
        metrics['real_dists'] = []
        metrics['dists'] = []
        return position_loss, None, None, None, None, metrics

    # As a metric, keep track of the bin distance using linear assignment. First compute it with 5 systems
    mean_dist_expanded, hr_0_expanded, hr_1_expanded, _, _, _ = compute_metrics_ijks(actual_ijks,
                                                                                     predicted_ijks_expanded)
    metrics['mean_dist_5'] = mean_dist_expanded
    metrics['hr_0_5'] = hr_0_expanded
    metrics['hr_1_5'] = hr_1_expanded

    # Then again, with the right amount
    if use_threshold:
        predicted_ijks = predicted_ijks_expanded
    else:
        predicted_ijks = predicted_ijks_expanded[:len(filtered_transforms)]
    mean_dist, hr_0, hr_1, dists, row_ind, col_ind = compute_metrics_ijks(actual_ijks, predicted_ijks)
    metrics['mean_dist'] = mean_dist
    metrics['hr_0'] = hr_0
    metrics['hr_1'] = hr_1
    metrics['dists'] = dists

    actual_distances = []
    predicted_probas = []
    predicted_rz_angle = []
    predicted_rz_norm = []
    predicted_theta_angle = []
    predicted_theta_norm = []
    selected_ijks = predicted_ijks[row_ind]
    for index, (i, j, k) in enumerate(selected_ijks):
        # Extract the predicted vector at this location
        predicted_proba = prediction_np[0, 0, i, j, k]
        predicted_probas.append(predicted_proba)

        ground_truth_translation = filtered_transforms[col_ind[index]][1]
        predicted_offset = prediction_np[0, 1:4, i, j, k]
        predicted_position = predicted_offset + np.asarray([bin_x[i], bin_y[j], bin_z[k]])
        distance = np.linalg.norm(ground_truth_translation - predicted_position)
        actual_distances.append(distance)

        # Get gt_rotation and metrics on the angles
        ground_truth_rotation = filtered_transforms[col_ind[index]][2]
        rz, theta = rotation_to_supervision(ground_truth_rotation)
        predicted_rz = prediction_np[0, 4:7, i, j, k]
        rz_norm = np.linalg.norm(predicted_rz)
        dot_product = np.dot(predicted_rz / rz_norm, rz)
        rz_angle = np.arccos(dot_product)

        # Following AF2 and to avoid singularities, we frame the prediction of an angle as a regression task in the plane.
        # We turn our angle into a unit vector of R2, push predicted norm to and penalize dot product to ground truth
        vec_angle = [np.cos(theta), np.sin(theta)]
        predicted_theta = prediction_np[0, 7:9, i, j, k]
        theta_norm = np.linalg.norm(predicted_theta)
        dot_product = np.dot(predicted_theta / theta_norm, vec_angle)
        theta_angle = np.arccos(dot_product)

        predicted_rz_angle.append(rz_angle)
        predicted_rz_norm.append(rz_norm)
        predicted_theta_angle.append(theta_angle)
        predicted_theta_norm.append(theta_norm)

    # This only makes a difference when we overpredicted.
    # Extract only the right predictions
    if use_threshold:
        overpredictions = len(predicted_ijks) - len(filtered_transforms)
        if overpredictions > 0:
            useless_ijks = np.delete(predicted_ijks, row_ind, axis=0)
            actual_distances += [20 for _ in range(overpredictions)]
            predicted_probas += [prediction_np[0, 0, i, j, k] for (i, j, k) in useless_ijks]
            predicted_rz_angle += [1000 for _ in range(overpredictions)]
            predicted_rz_norm += [1000 for _ in range(overpredictions)]
            predicted_theta_angle += [1000 for _ in range(overpredictions)]
            predicted_theta_norm += [1000 for _ in range(overpredictions)]
    metrics['real_dists'] = actual_distances
    metrics['probas'] = predicted_probas
    metrics['rz_angle'] = predicted_rz_angle
    metrics['rz_norm'] = predicted_rz_norm
    metrics['theta_angle'] = predicted_theta_angle
    metrics['theta_norm'] = predicted_theta_norm

    metrics['nano_classifs'] = list()
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
            right_classif = nano == (vector_pose[8] > 0.5).item()
            # print(nano, vector_pose[8], nano_loss)
        else:
            nano_loss = torch.zeros(1)
            right_classif = 0
        metrics['nano_classifs'].append(right_classif)

        offset_losses.append(offset_loss)
        rz_losses.append(rz_loss)
        angle_losses.append(angle_loss)
        nano_losses.append(nano_loss)

    offset_loss = torch.mean(torch.stack(offset_losses))
    rz_loss = torch.mean(torch.stack(rz_losses))
    angle_loss = torch.mean(torch.stack(angle_losses))
    nano_loss = torch.mean(torch.stack(nano_losses))
    return position_loss, offset_loss, rz_loss, angle_loss, nano_loss, metrics
