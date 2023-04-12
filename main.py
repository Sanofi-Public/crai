import sys, os

import time

import numpy as np
import scipy

import torch
from torch.utils.tensorboard import SummaryWriter

import pymol
import pymol2

pymol.invocation.parse_args(['pymol', '-q'])  # optional, for quiet flag
pymol2.SingletonPyMOL().start()

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, ''))

from load_data.ABDataset import ABDataset
from learning.model import UnetModel
from utils.learning_utils import weighted_dice_loss, weighted_ce_loss, get_split_dataloaders


def train(model, device, optimizer, loss_fn, loader, writer, n_epochs=10, val_loader=None, accumulated_batch=1):
    time_init = time.time()

    for epoch in range(n_epochs):
        for step, (name, input_grid, output_grid) in enumerate(loader):
            if name[0] == 'failed' or isinstance(input_grid, list):
                continue
            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)

            prediction = model(input_grid)
            loss = loss_fn(prediction, output_grid)
            loss.backward()

            # Accumulated gradients
            if not step % accumulated_batch:
                optimizer.step()
                model.zero_grad()

            if not step % 20:
                step_total = len(loader) * epoch + step
                print("GT        : ", torch.mean(output_grid[0], dim=(1, 2, 3)).data)
                print("predicted : ", torch.mean(prediction[0], dim=(1, 2, 3)).data)
                eluded_time = time.time() - time_init
                print(f"Epoch : {epoch} ; step : {step} ; loss : {loss.item():.5f} ; time : {eluded_time:.1f}")
                writer.add_scalar('train_loss', loss.item(), step_total)
        if val_loader is not None:
            print("validation")
            losses = validate(model=model, device=device, loss_fn=loss_fn, loader=val_loader)
            mean_val_loss = np.mean(losses)
            print(f'Validation loss ={mean_val_loss}')
            writer.add_scalar('loss_val', mean_val_loss, epoch)
        if epoch == 99:
            model.cpu()
            model_path = os.path.join("saved_models", 'fifth_100.pth')
            torch.save(model.state_dict(), model_path)
            model.to(device)


def validate(model, device, loss_fn, loader):
    time_init = time.time()
    losses = list()
    with torch.no_grad():
        for step, (name, input_grid, output_grid) in enumerate(loader):
            if name[0] == 'failed' or isinstance(input_grid, list):
                continue

            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)

            prediction = model(input_grid)
            loss = loss_fn(prediction, output_grid)

            losses.append(loss.item())
            if not step % 20:
                error_norm = torch.sqrt(loss).item()
                print(f"step : {step} ; loss : {loss.item():.5f} ; time : {time.time() - time_init:.1f}")
    return losses


def local_loss_fn(x, y):
    # return weighted_ce_loss(x,y)
    # return weighted_dice_loss(x,y)
    categorical_slices_x = x[..., :3, :, :, :]
    categorical_slices_y = y[..., :3, :, :, :]
    weight = 1 / (torch.mean(categorical_slices_y, dim=(-3, -2, -1)) + 1e-4)
    weight = weight / torch.sum(weight)
    weight = weight[..., None, None, None]
    loss = weighted_dice_loss(categorical_slices_x, categorical_slices_y) \
           + weighted_ce_loss(categorical_slices_x, categorical_slices_y, weight=weight)
    if x.shape[-4] > 3:
        mse_slices_x = x[..., -2:, :, :, :]
        mse_slices_y = y[..., -2:, :, :, :]
        loss += torch.nn.functional.mse_loss(mse_slices_x, mse_slices_y)
    return loss


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", default='default')
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name

    # Setup learning
    data_root = "data/pdb_em"
    csv_to_read = "data/final.csv"
    # csv_to_read = "data/reduced_final.csv"
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"logs/{model_name}")
    model_path = os.path.join("saved_models", f'{model_name}.pth')
    gpu_number = args.gpu
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'

    # Setup data
    rotate = True
    return_sdf = True
    # num_workers = 0
    num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw
    ab_dataset = ABDataset(data_root=data_root,
                           csv_to_read=csv_to_read,
                           rotate=rotate,
                           return_sdf=return_sdf)
    train_loader, val_loader, _ = get_split_dataloaders(dataset=ab_dataset,
                                                        shuffle=True,
                                                        num_workers=num_workers)

    # Learning hyperparameters
    n_epochs = 100
    loss_fn = local_loss_fn
    accumulated_batch = 5
    model = UnetModel(predict_mse=return_sdf).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Train
    train(model=model, device=device, loss_fn=loss_fn, loader=train_loader,
          optimizer=optimizer, writer=writer, n_epochs=n_epochs, val_loader=val_loader,
          accumulated_batch=accumulated_batch)
    model.cpu()
    torch.save(model.state_dict(), model_path)
