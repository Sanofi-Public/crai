import os
import sys

import glob
import numpy as np
import pickle
import time
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from learning.loss_and_metrics import coords_loss
from learning.SimpleUnet import SimpleHalfUnetModel
from learning.train_coords import validate, dump_log
from load_data.ABDataset import ABDataset
from utils.learning_utils import setup_learning
from utils.python_utils import mini_hash


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


def relog(model, model_name, val_loader, gpu=0):
    writer, _, device = setup_learning(model_name=model_name,
                                       gpu_number=gpu)
    weights = weights_from_name(model_name)
    for epoch, weight in weights:
        time_init = time.time()
        model.load_state_dict(torch.load(weight))
        model = model.to(device)
        to_log = validate(model=model, device=device, loader=val_loader)
        val_loss = to_log["loss"]
        dump_log(writer, epoch, to_log, prefix='full_val_')
        print(f'Validation loss ={val_loss}, epoch : {epoch}, time : {time.time() - time_init}')
        writer.flush()


def validate_detailed(model, model_name, loader, outname, gpu=0, use_threshold=False, use_pd=False):
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    weights_path = f"../saved_models/{model_name}.pth"
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
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
            position_loss, offset_loss, rz_loss, angle_loss, nano_loss, metrics = coords_loss(prediction, comp,
                                                                                              classif_nano=False,
                                                                                              ot_weight=0,
                                                                                              use_threshold=use_threshold,
                                                                                              use_pd=use_pd)
            dict_res[name] = metrics
            if offset_loss is not None:
                loss = position_loss + offset_loss + rz_loss + angle_loss
                losses.append([loss.item(),
                               position_loss.item(),
                               offset_loss.item(),
                               rz_loss.item(),
                               angle_loss.item(),
                               nano_loss.item(),
                               ])
            if not step % 100:
                print(f"step : {step} ; loss : {loss.item():.5f} ; time : {time.time() - time_init:.1f}")
        pickle.dump(dict_res, open(outname, 'wb'))
    all_dists_flat = np.concatenate([met['real_dists'] for met in dict_res.values()])
    print('Uncapped', np.mean(all_dists_flat))
    all_dists_flat[all_dists_flat > 20] = 20
    print('Capped', np.mean(all_dists_flat))
    losses = np.array(losses)
    losses = np.mean(losses, axis=0)
    print(losses)
    return losses


# Setup data
def get_loader(sorted=False, split='val', nano=False, normalize='max', num_workers=4):
    csv_val = f"../data/{'nano_' if nano else ''}csvs/{'sorted_' if sorted else ''}chunked_{split}.csv"
    all_system_val = f"../data/{'nano_' if nano else ''}csvs/{'sorted_' if sorted else ''}filtered_{split}.csv"
    ab_dataset = ABDataset(all_systems=all_system_val, csv_to_read=csv_val,
                           rotate=False, crop=0, full=True, normalize=normalize)
    ab_loader = torch.utils.data.DataLoader(dataset=ab_dataset, collate_fn=lambda x: x[0], num_workers=num_workers)
    return ab_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", default='default')
    parser.add_argument("--nano", action='store_true', default=False)
    parser.add_argument("--sorted", action='store_true', default=False)
    parser.add_argument("--thresh", action='store_true', default=False)
    parser.add_argument("--pd", action='store_true', default=False)
    parser.add_argument("--split", default='val')
    parser.add_argument("-norm", "--normalize", default='max', help='one of None, max, centile')
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Learning hyperparameters
    model = SimpleHalfUnetModel(in_channels=1,
                                model_depth=4,
                                classif_nano=args.nano,
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

    # loader = get_loader(sorted=args.sorted, nano=args.nano, normalize=args.normalize)
    # relog(model=model, model_name=args.model_name, val_loader=loader, gpu=0)

    # VALIDATE DETAILED
    loader = get_loader(sorted=args.sorted, split=args.split, nano=args.nano, normalize=args.normalize)
    # Include all information and add hash for simpler bookkeeping
    outstring = (f"{args.model_name}_{args.nano}_{args.sorted}_{args.split}{'_thresh' if args.thresh else ''}"
                 f"{'_pd' if args.pd else ''}.p")
    outname = f"../outfiles/out_{mini_hash(outstring)}_{outstring}"
    validate_detailed(model=model, model_name=args.model_name, loader=loader, outname=outname,
                      gpu=args.gpu, use_threshold=args.thresh, use_pd=args.pd)
