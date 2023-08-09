import os
import sys

import time

import numpy as np
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
from learning.loss_and_metrics import coords_loss
from learning.SimpleUnet import SimpleHalfUnetModel
from utils.learning_utils import setup_learning


def dump_log(writer, epoch, to_log, prefix=''):
    for name, value in to_log.items():
        writer.add_scalar(f'{prefix}{name}', value, epoch)


def train(model, loader, optimizer, n_epochs=10, device='cpu', classif=False,
          loc_weight=1., other_weight=0.2,
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
            position_loss, offset_loss, rz_loss, angle_loss, nano_loss, metrics = coords_loss(prediction, comp,
                                                                                              classif_nano=classif,
                                                                                              ot_weight=args.otw)
            if offset_loss is None:
                loss = position_loss
            else:
                loss = loc_weight * position_loss + \
                       other_weight * (0.3 * offset_loss + rz_loss + angle_loss + nano_loss)
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
                    position_dist = metrics['mean_dist']
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
            to_log = validate(model=model, device=device, loader=val_loader, classif_nano=classif)
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
            to_log = validate(model=model, device=device, loader=val_loader_full, classif_nano=classif)
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
            to_log = validate(model=model, device=device, loader=nano_loader, classif_nano=classif)
            val_loss = to_log["loss"]
            print(f'Validation loss ={val_loss}')
            dump_log(writer, epoch, to_log, prefix='nano_val_')
            if val_loader_full is None and val_loss < best_mean_val_loss:
                best_mean_val_loss = val_loss
                best_model_path = f'{save_path}_{epoch}.pth'
                model.cpu()
                torch.save(model.state_dict(), best_model_path)
                model.to(device)


def validate(model, device, loader, classif_nano=True):
    time_init = time.time()
    losses = list()
    all_real_dists = list()
    all_nano_classifs = list()
    with torch.no_grad():
        for step, (name, comp) in enumerate(loader):
            if comp is None:
                continue
            input_tensor = torch.from_numpy(comp.input_tensor[None, ...]).to(device)
            prediction = model(input_tensor)
            position_loss, offset_loss, rz_loss, angle_loss, nano_loss, metrics = coords_loss(prediction, comp,
                                                                                              classif_nano=classif_nano)

            if offset_loss is not None:
                position_dist = metrics['mean_dist']
                real_dists = metrics['real_dists']
                nano_classifs = metrics['nano_classifs']
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
                all_nano_classifs.extend(nano_classifs)

            if not step % 100:
                print(f"step : {step} ; loss : {loss.item():.5f} ; time : {time.time() - time_init:.1f}")
    losses = np.array(losses)
    losses = np.mean(losses, axis=0)
    all_real_dists = np.asarray(all_real_dists)
    hr_6 = sum(all_real_dists <= 6) / len(all_real_dists)
    mean_real_dist = np.mean(all_real_dists)
    all_real_dists[all_real_dists >= 20] = 20
    mean_real_dist_capped = np.mean(all_real_dists)
    nano_classif_rate = np.mean(np.asarray(all_nano_classifs))
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
              "nano_classif_rate": nano_classif_rate,
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
    parser.add_argument("--use_fabs", action='store_false', default=True)
    parser.add_argument("--use_nano", action='store_false', default=True)
    parser.add_argument("-norm", "--normalize", default='max', help='one of None, max, centile')
    parser.add_argument("--sorted", action='store_true', default=False)
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--otw", type=float, default=1.)
    parser.add_argument("--loc_weight", type=float, default=1.)
    parser.add_argument("--other_weight", type=float, default=0.2)
    parser.add_argument("--agg_grads", type=int, default=4)
    parser.add_argument("--crop", type=int, default=3)
    args = parser.parse_args()

    writer, save_path, device = setup_learning(model_name=args.model_name,
                                               gpu_number=args.gpu)

    # Setup data

    # num_workers = 0
    num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw
    all_systems_train = []
    csv_train = []
    if args.use_fabs:
        all_systems_train.append(f"../data/csvs/{'sorted_' if args.sorted else ''}filtered_train.csv")
        csv_train.append(f"../data/csvs/{'sorted_' if args.sorted else ''}chunked_train.csv")
    if args.use_nano:
        all_systems_train.append(f"../data/nano_csvs/{'sorted_' if args.sorted else ''}filtered_train.csv")
        csv_train.append(f"../data/nano_csvs/{'sorted_' if args.sorted else ''}chunked_train.csv")

    # DEBUG
    # csv_train=['../data/csvs/reduced.csv']
    train_ab_dataset = ABDataset(csv_to_read=csv_train, all_systems=all_systems_train,
                                 rotate=args.rotate, crop=args.crop, full=args.train_full, normalize=args.normalize)
    train_loader = DataLoader(dataset=train_ab_dataset, worker_init_fn=np.random.seed,
                              shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    # Test loss
    # fake_out = torch.randn((1, 9, 23, 28, 19))
    # fake_out[0, 0, ...] = torch.sigmoid(fake_out[0, 0, ...])
    # coords_loss(fake_out, train_ab_dataset[0][1])
    # sys.exit()

    if args.use_fabs:
        csv_val = f"../data/csvs/{'sorted_' if args.sorted else ''}chunked_val.csv"
        all_system_val = f"../data/csvs/{'sorted_' if args.sorted else ''}filtered_val.csv"
        # val_ab_dataset = ABDataset(all_systems=all_system_val,csv_to_read=csv_val,
        #                            rotate=False, crop=0, full=False, normalize=args.normalize)
        # val_loader = DataLoader(dataset=val_ab_dataset, collate_fn=lambda x: x[0], num_workers=num_workers)
        val_loader = None
        val_ab_dataset_full = ABDataset(all_systems=all_system_val, csv_to_read=csv_val,
                                        rotate=False, crop=0, full=True, normalize=args.normalize)
        val_loader_full = DataLoader(dataset=val_ab_dataset_full, collate_fn=lambda x: x[0], num_workers=num_workers)
    else:
        val_loader = None
        val_loader_full = None
    if args.use_nano:
        csv_val_nano = f"../data/nano_csvs/{'sorted_' if args.sorted else ''}chunked_val.csv"
        all_system_val_nano = f"../data/nano_csvs/{'sorted_' if args.sorted else ''}filtered_val.csv"
        val_ab_dataset_nano_full = ABDataset(all_systems=all_system_val_nano, csv_to_read=csv_val_nano,
                                             rotate=False, crop=0, full=True, normalize=args.normalize)
        val_loader_nano_full = DataLoader(dataset=val_ab_dataset_nano_full, collate_fn=lambda x: x[0],
                                          num_workers=num_workers)
    else:
        val_loader_nano_full = None

    # Learning hyperparameters
    classif = args.use_nano and args.use_fabs
    n_epochs = 1000
    accumulated_batch = args.agg_grads
    model = SimpleHalfUnetModel(in_channels=1,
                                classif_nano=classif,
                                out_channels=9,
                                model_depth=4,
                                num_convs=3,
                                max_decode=2,
                                num_feature_map=32)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train(model=model, loader=train_loader, optimizer=optimizer, n_epochs=n_epochs, device=device, classif=classif,
          loc_weight=args.loc_weight, other_weight=args.other_weight,
          val_loader=val_loader, val_loader_full=val_loader_full, nano_loader=val_loader_nano_full,
          writer=writer, accumulated_batch=accumulated_batch, save_path=save_path)
