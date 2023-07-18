import os
import sys

import time
import glob
import pickle
import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from load_data.ABDataset import ABDataset
from learning.SimpleUnet import SimpleHalfUnetModel
from utils.learning_utils import setup_learning
from learning.train_coords import coords_loss, validate, dump_log


def weights_from_name(name):
    hits = glob.glob(f"../saved_models/{name}*")
    weights = [(hit.split('_')[-1].split('.')[0], hit) for hit in hits]
    # Necessary to remove 'last' and 'best' models. Could be done with fancier regex
    filtered_weights = []
    for epoch, weight in weights:
        try:
            filtered_weights.append((int(epoch), weight))
        except ValueError:
            pass
    weights = sorted(filtered_weights, key=lambda x: x[0])
    weights.append((350, f"../saved_models/{name}_last.pth"))
    return weights


def relog(model, device, weights, val_loader, writer):
    for epoch, weight in weights:
        time_init = time.time()
        model.load_state_dict(torch.load(weight))
        model = model.to(device)
        to_log = validate(model=model, device=device, loader=val_loader)
        val_loss = to_log["loss"]
        dump_log(writer, epoch, to_log, prefix='full_val_')
        print(f'Validation loss ={val_loss}, epoch : {epoch}, time : {time.time() - time_init}')
        writer.flush()


def validate_detailed(model, device, loader, outname):
    time_init = time.time()
    losses = list()
    dict_res = {}
    with torch.no_grad():
        for step, (name, comp) in enumerate(loader):
            if comp is None:
                dict_res[name] = None
                continue
            input_tensor = torch.from_numpy(comp.input_tensor[None, ...]).to(device)
            prediction = model(input_tensor)
            position_loss, offset_loss, rz_loss, angle_loss, nano_loss, metrics = coords_loss(prediction, comp)
            position_dist = metrics['mean_dist']
            real_dists = metrics['real_dists']
            all_dists = metrics['dists']
            if offset_loss is not None:
                loss = position_loss + offset_loss + rz_loss + angle_loss
                losses.append([loss.item(),
                               position_loss.item(),
                               offset_loss.item(),
                               rz_loss.item(),
                               angle_loss.item(),
                               nano_loss.item(),
                               position_dist
                               ])

                dict_res[name] = all_dists, real_dists
            else:
                dict_res[name] = None
            if not step % 100:
                print(f"step : {step} ; loss : {loss.item():.5f} ; time : {time.time() - time_init:.1f}")
        pickle.dump(dict_res, open(outname, 'wb'))
    return losses


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", default='default')
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Setup data
    rotate = False
    crop = 0
    # num_workers = 0
    num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw
    # csv_to_read = "../data/csvs/chunked_val_reduced.csv"
    csv_to_read = "../data/csvs/chunked_val.csv"
    val_ab_dataset = ABDataset(csv_to_read=csv_to_read,
                               rotate=rotate,
                               crop=crop,
                               full=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_ab_dataset,
                                             collate_fn=lambda x: x[0],
                                             num_workers=num_workers)
    # Learning hyperparameters
    model = SimpleHalfUnetModel(in_channels=1,
                                model_depth=4,
                                num_convs=3,
                                max_decode=2,
                                num_feature_map=32)

    # RELOG
    # for name in ['big_train', 'multi_train', 'big_train_crop', 'big_train_full_df', 'big_train_gamma',
    #              'big_train_normalize', 'focal', 'recrop', 'rebig']:
    #     writer, _, device = setup_learning(model_name=name,
    #                                        gpu_number=args.gpu)
    #     weights = weights_from_name(name)
    #     relog(model=model, device=device, weights=weights, writer=writer, val_loader=val_loader)

    writer, _, device = setup_learning(model_name=args.model_name,
                                       gpu_number=args.gpu)
    weights = weights_from_name(args.model_name)
    relog(model=model, device=device, weights=weights, writer=writer, val_loader=val_loader)

    # VALIDATE DETAILED
    # device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    # # model_name = "multi_train_339"
    # # model_name = "multi_train_861"
    # model_name = "big_train_gamma_last"
    # # model_name=args.model_name
    # weights_path = f"../saved_models/{model_name}.pth"
    # model.load_state_dict(torch.load(weights_path))
    # model = model.to(device)
    # outname = f"out_{model_name}.p"
    # validate_detailed(model=model, device=device, loader=val_loader, outname=outname)
