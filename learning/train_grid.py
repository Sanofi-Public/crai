import os
import sys

import numpy as np
import time
import torch
import pymol
import pymol2

pymol.invocation.parse_args(['pymol', '-q'])  # optional, for quiet flag
pymol2.SingletonPyMOL().start()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from load_data.ABDataset import ABDataset
from learning.Unet import UnetModel
from utils.learning_utils import weighted_dice_loss, weighted_ce_loss, get_split_dataloaders, setup_learning


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


def loop_fn(model, device, complex):
    input_tensor = torch.from_numpy(complex.input_tensor[None, ...]).to(device)
    target_tensor = torch.from_numpy(complex.target_tensor[None, ...]).to(device)
    prediction = model(input_tensor)
    loss = loss_fn(prediction, target_tensor)
    return loss


def train(model, device, optimizer, loader, loop_fn,
          writer=None, n_epochs=10, val_loader=None, accumulated_batch=1, save_path=''):
    best_mean_val_loss = 10000.
    last_model_path = f'{save_path}_last.pth'
    best_model_path = f'{save_path}_best.pth'
    time_init = time.time()
    for epoch in range(n_epochs):
        for step, (name, complex) in enumerate(loader):
            if complex is None:
                continue

            loss = loop_fn(model, device, complex)
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

        model.cpu()
        torch.save(model.state_dict(), last_model_path)
        model.to(device)

        if val_loader is not None:
            print("validation")
            losses = validate(model=model, device=device, loop_fn=loop_fn, loader=val_loader)
            mean_val_loss = np.mean(losses)
            print(f'Validation loss ={mean_val_loss}')
            writer.add_scalar('loss_val', mean_val_loss, epoch)
            # Model checkpointing
            if mean_val_loss < best_mean_val_loss:
                best_mean_val_loss = mean_val_loss
                model.cpu()
                torch.save(model.state_dict(), best_model_path)
                model.to(device)


def validate(model, device, loop_fn, loader):
    time_init = time.time()
    losses = list()
    with torch.no_grad():
        for step, (name, complex) in enumerate(loader):
            if complex is None:
                continue
            loss = loop_fn(model, device, complex)
            losses.append(loss.item())
            if not step % 20:
                print(f"step : {step} ; loss : {loss.item():.5f} ; time : {time.time() - time_init:.1f}")
    return losses


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", default='default')
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    writer, best_model_path, last_model_path, device = setup_learning(model_name=args.model_name,
                                                                      gpu_number=args.gpu, )

    # Setup data
    rotate = True
    return_sdf = True
    num_workers = 0
    # num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw
    data_root = "../data/pdb_em"
    csv_to_read = "../data/reduced_final.csv"
    # csv_to_read = "data/final.csv"
    ab_dataset = ABDataset(data_root=data_root,
                           csv_to_read=csv_to_read,
                           rotate=rotate,
                           return_sdf=return_sdf)
    train_loader, val_loader, _ = get_split_dataloaders(dataset=ab_dataset,
                                                        shuffle=True,
                                                        num_workers=num_workers,
                                                        collate_fn=lambda x: x[0])
    # Learning hyperparameters
    n_epochs = 700
    loss_fn = local_loss_fn
    accumulated_batch = 5
    model = UnetModel(predict_mse=return_sdf,
                      out_channels_decoder=128,
                      num_feature_map=24, )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Train
    train(model=model, device=device, loop_fn=loop_fn, loader=train_loader,
          optimizer=optimizer, writer=writer, n_epochs=n_epochs, val_loader=val_loader,
          accumulated_batch=accumulated_batch, save_path=best_model_path)
    model.cpu()
    torch.save(model.state_dict(), last_model_path)
