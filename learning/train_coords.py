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
from learning.SimpleUnet import SimpleHalfUnetModel
from utils.learning_utils import get_split_dataloaders, rotation_to_supervision
from utils.learning_utils import weighted_bce, weighted_dice_loss, weighted_focal_loss
from learning.train_functions import setup_learning


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
    # position_loss = weighted_bce(prediction[0, 0, ...], BCE_target, weights=[1, 1000])
    position_loss = weighted_focal_loss(prediction[0, 0, ...],
                                        BCE_target,
                                        weights=[1, 30])

    # And as a metric, keep track of the bin distance
    amax = torch.argmax(prediction[0, 0, ...]).cpu().detach().numpy()
    pred_x, pred_y, pred_z = np.unravel_index(amax, pred_shape)
    position_dist = np.linalg.norm(np.array([pred_x, pred_y, pred_z]) - np.array([position_x, position_y, position_z]))
    position_dist = float(position_dist)

    # Extract the predicted vector at this location
    vector_pose = prediction[0, 1:, position_x, position_y, position_z]

    # Get the offset from the corner prediction loss
    offset_x = complex.translation[0] - bin_x[position_x]
    offset_y = complex.translation[1] - bin_y[position_y]
    offset_z = complex.translation[2] - bin_z[position_z]
    gt_offset = torch.tensor([offset_x, offset_y, offset_z], device=device, dtype=torch.float)
    offset_loss = torch.nn.MSELoss()(vector_pose[:3], gt_offset)

    # Get the right pose. For that get the rotation supervision as a R3 vector and an angle.
    # We will penalise the R3 with its norm and it's dot product to ground truth
    rz, angle = rotation_to_supervision(complex.rotation)
    rz = torch.tensor(rz, device=device, dtype=torch.float)
    predicted_rz = vector_pose[3:6]
    rz_norm = torch.norm(predicted_rz)
    rz_loss = 1 - torch.dot(predicted_rz / rz_norm, rz) + (rz_norm - 1) ** 2

    # Following AF2 and to avoid singularities, we frame the prediction of an angle as a regression task in the plane.
    # We turn our angle into a unit vector of R2, push predicted norm to and penalize dot product to ground truth
    vec_angle = [np.cos(angle), np.sin(angle)]
    vec_angle = torch.tensor(vec_angle, device=device, dtype=torch.float)
    predicted_angle = vector_pose[6:]
    angle_norm = torch.norm(predicted_angle)
    angle_loss = 1 - torch.dot(predicted_angle / angle_norm, vec_angle) + (angle_norm - 1) ** 2

    return position_loss, offset_loss, rz_loss, angle_loss, position_dist


def train(model, device, optimizer, loader,
          writer=None, n_epochs=10, val_loader=None, accumulated_batch=1, save_path=''):
    best_mean_val_loss = 10000.
    last_model_path = f'{save_path}_last.pth'
    time_init = time.time()
    for epoch in range(n_epochs):
        for step, (name, comp) in enumerate(loader):
            if comp is None:
                continue
            input_tensor = torch.from_numpy(comp.input_tensor[None, ...]).to(device)
            prediction = model(input_tensor)
            try:
                position_loss, offset_loss, rz_loss, angle_loss, position_dist = coords_loss(prediction, comp)
            except IndexError:
                print("skipped one")
                continue
            loss = position_loss + 0.2 * (0.3 * offset_loss + rz_loss + angle_loss)
            # loss = position_loss +  offset_loss + rz_loss + angle_loss
            loss.backward()

            # Accumulated gradients
            if not step % accumulated_batch:
                optimizer.step()
                model.zero_grad()

            if not step % 20:
                step_total = len(loader) * epoch + step
                eluded_time = time.time() - time_init
                print(f"Epoch : {epoch} ; step : {step} ; loss : {loss.item():.5f} ; time : {eluded_time:.1f}")
                writer.add_scalar('train_loss', loss.item(), step_total)
                writer.add_scalar('train_position_loss', position_loss.item(), step_total)
                writer.add_scalar('train_position_distance', position_dist, step_total)
                writer.add_scalar('train_offset_loss', offset_loss.item(), step_total)
                writer.add_scalar('train_rz_loss', rz_loss.item(), step_total)
                writer.add_scalar('train_angle_loss', angle_loss.item(), step_total)

        if epoch == n_epochs - 1:
            model.cpu()
            torch.save(model.state_dict(), last_model_path)
            model.to(device)

        if val_loader is not None:
            print("validation")
            losses = validate(model=model, device=device, loader=val_loader)
            losses = np.array(losses)
            val_loss, position_loss, offset_loss, rz_loss, angle_loss, position_dist = np.mean(losses, axis=0)
            print(f'Validation loss ={val_loss}')
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_position_loss', position_loss, epoch)
            writer.add_scalar('val_position_distance', position_dist, epoch)
            writer.add_scalar('val_offset_loss', offset_loss, epoch)
            writer.add_scalar('val_rz_loss', rz_loss, epoch)
            writer.add_scalar('val_angle_loss', angle_loss, epoch)
            # Model checkpointing
            if val_loss < best_mean_val_loss:
                best_mean_val_loss = val_loss
                best_model_path = f'{save_path}_{epoch}.pth'
                model.cpu()
                torch.save(model.state_dict(), best_model_path)
                model.to(device)


def validate(model, device, loader):
    time_init = time.time()
    losses = list()
    with torch.no_grad():
        for step, (name, comp) in enumerate(loader):
            if comp is None:
                continue
            input_tensor = torch.from_numpy(comp.input_tensor[None, ...]).to(device)
            prediction = model(input_tensor)
            try:
                position_loss, offset_loss, rz_loss, angle_loss, position_dist = coords_loss(prediction, comp)
            except IndexError:
                print("skipped one")
                continue
            loss = position_loss + offset_loss + rz_loss + angle_loss
            losses.append([loss.item(),
                           position_loss.item(),
                           offset_loss.item(),
                           rz_loss.item(),
                           angle_loss.item(),
                           position_dist
                           ])
            if not step % 20:
                print(f"step : {step} ; loss : {loss.item():.5f} ; time : {time.time() - time_init:.1f}")
    return losses


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", default='default')
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--agg_grads", type=int, default=20)
    args = parser.parse_args()

    writer, save_path, device = setup_learning(model_name=args.model_name,
                                               gpu_number=args.gpu)

    # Setup data
    rotate = True
    crop = 3
    # num_workers = 0
    num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw
    data_root = "../data/pdb_em"
    # csv_to_read = "../data/reduced_final.csv"
    csv_to_read = "../data/cleaned_final.csv"
    # csv_to_read = "../data/final.csv"
    ab_dataset = ABDataset(data_root=data_root,
                           csv_to_read=csv_to_read,
                           rotate=rotate,
                           crop=crop,
                           return_grid=False)
    train_loader, val_loader, _ = get_split_dataloaders(dataset=ab_dataset,
                                                        shuffle=True,
                                                        collate_fn=lambda x: x[0],
                                                        num_workers=num_workers)
    # # Test loss
    # fake_out = torch.randn((1, 9, 23, 28, 19))
    # fake_out[0, 0, ...] = torch.sigmoid(fake_out[0, 0, ...])
    # coords_loss(fake_out, ab_dataset[0][1])

    # Learning hyperparameters
    n_epochs = 1000
    accumulated_batch = args.agg_grads
    # model = HalfUnetModel(out_channels_decoder=128,
    #                       num_feature_map=24, )
    model = SimpleHalfUnetModel(in_channels=1,
                                model_depth=4,
                                num_convs=3,
                                max_decode=2,
                                num_feature_map=32)
    # model_path = "../saved_models/object_4_last.pth"
    # model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train(model=model, device=device, loader=train_loader,
          optimizer=optimizer, writer=writer, n_epochs=n_epochs, val_loader=val_loader,
          accumulated_batch=accumulated_batch, save_path=save_path)

    # ######################################
    # 
    # pl.seed_everything(seed, workers=True)
    # # init model
    # model = PIPModule(cfg)
    # tb_logger = TensorBoardLogger(save_dir=log_dir, version=name)
    # loggers = [tb_logger]
    # # init trainer
    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     devices=[args.gpu],
    #     max_epochs=200,
    #     logger=loggers,
    # )
    # # datamodule
    # datamodule = PLDataModule(PIPDataset, cfg.dataset.data_dir, cfg.loader.batch_size_train)
    # # train
    # trainer.fit(model, datamodule=datamodule)
    # # test
    # trainer.test(model, datamodule=datamodule)
