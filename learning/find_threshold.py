import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from load_data.ABDataset import ABDataset
from learning.SimpleUnet import SimpleHalfUnetModel
from utils.object_detection import nms
from utils.python_utils import mini_hash


def find_thresh(model, model_name, loader, gpu=0, use_pd=False, outname=None):
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    weights_path = f"../saved_models/{model_name}.pth"
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    dict_res = {}
    # threshs = [0.35]
    threshs = np.linspace(0., 1., num=50)
    all_results = np.empty((len(loader), len(threshs)))
    all_results.fill(np.nan)
    with (torch.no_grad()):
        for step, (name, comp) in enumerate(loader):
            input_tensor = torch.from_numpy(comp.input_tensor[None, ...]).to(device)
            prediction = model(input_tensor).cpu().numpy()
            prediction = prediction[0, 0]
            true_n = len(comp.transforms)
            if not step % 50:
                print(f"Done {step}")
            for i, thresh in enumerate(threshs):
                ijk_s = nms(prediction, n_objects=None, thresh=thresh, use_pd=use_pd)
                predicted_n = len(ijk_s)
                error = abs(true_n - predicted_n)
                all_results[step, i] = error
                dict_res[name] = error
    all_values = np.nanmean(all_results, axis=0)
    # In practice there are no nans
    # all_nans = np.sum(np.isnan(all_results), axis=0)
    for thresh, value in zip(threshs, all_values):
        print(f"{thresh:.2f}, {value:.2f}")

    if outname is not None:
        print(threshs.shape)
        print(all_values.shape)
        np.save(outname, np.stack((threshs, np.squeeze(all_values))))
    # for k, v in dict_res.items():
    #     print(k, v)
    # print(f"hit rate : {np.sum([x == 0 for x in dict_res.values()]) / len(dict_res)}")
    return dict_res


def line_plot(npy_name, label, color=None):
    a = np.load(npy_name)
    threshs, all_values = a[0, :], a[1, :]
    all_values = np.clip(all_values, 0, 1)
    ax = plt.gca()
    ax.plot(threshs, all_values, label=label, color=color, linewidth=2)


def plot_thresh():
    # f"out_{args.model_name}_{args.nano}_{args.sorted}_{args.split}_{args.pd}.npy"
    palette = sns.color_palette("Paired")
    # palette = sns.color_palette(n_colors=8)
    line_plot("out_87_fr_final_last_val_True.npy",
              "PD", color=palette[0])
    line_plot("out_96_fr_final_last_val_False.npy",
              "Margin", color=palette[1])
    line_plot("out_59_fs_final_last_val_True.npy",
              "Sorted PD", color=palette[2])
    line_plot("out_84_fs_final_last_val_False.npy",
              "Sorted Margin", color=palette[3])
    line_plot("out_76_nr_final_last_val_True.npy",
              "Nano PD", color=palette[4])
    line_plot("out_98_nr_final_last_val_False.npy",
              "Nano Margin", color=palette[5])
    line_plot("out_28_ns_final_last_val_True.npy",
              "Nano Sorted PD", color=palette[6])
    line_plot("out_32_ns_final_last_val_False.npy",
              "Nano Sorted Margin", color=palette[7])
    # Just to see what test looks like
    # line_plot("out_75_nr_final_last_test_True.npy",
    #           "Nano PD test", color=palette[6])
    # line_plot("out_53_nr_final_last_test_False.npy",
    #           "Nano Margin test", color=palette[7])
    plt.legend()
    plt.xlabel("Probability threshold")
    plt.ylabel("Mean error in the number of predictions")
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--nano", action='store_true', default=False)
    parser.add_argument("--sorted", action='store_true', default=False)
    parser.add_argument("--pd", action='store_true', default=False)
    parser.add_argument("--split", default='val')
    parser.add_argument("--nw", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()


    # Setup data
    def get_loader(sorted=False, split='val', nano=False, num_workers=4):
        csv_val = f"../data/{'nano_' if nano else ''}csvs/{'sorted_' if sorted else ''}chunked_{split}.csv"
        all_system_val = f"../data/{'nano_' if nano else ''}csvs/{'sorted_' if sorted else ''}filtered_{split}.csv"
        ab_dataset = ABDataset(all_systems=all_system_val, csv_to_read=csv_val,
                               rotate=False, crop=0, full=True, normalize='max')
        ab_loader = torch.utils.data.DataLoader(dataset=ab_dataset, collate_fn=lambda x: x[0], num_workers=num_workers)
        return ab_loader


    # Learning hyperparameters
    model = SimpleHalfUnetModel(in_channels=1,
                                model_depth=4,
                                classif_nano=args.nano,
                                num_convs=3,
                                max_decode=2,
                                num_feature_map=32)
    loader = get_loader(sorted=args.sorted, split=args.split, nano=args.nano)
    model_name = f"{'n' if args.nano else 'f'}{'s' if args.sorted else 'r'}_final_last"
    outstring = f"{model_name}_val_{args.pd}.npy"
    outname = f"out_{mini_hash(outstring)}_{outstring}"
    # find_thresh(model=model, model_name=model_name, loader=loader, gpu=args.gpu, use_pd=args.pd, outname=outname)
    plot_thresh()

# VANILLA NMS
# 0.12, 0.23, 0
# 0.14, 0.23, 0
# 0.16, 0.23, 0
# 0.18, 0.23, 0
# 0.20, 0.23, 0
# 0.22, 0.23, 0
# 0.24, 0.22, 0
# 0.27, 0.22, 0
# 0.29, 0.22, 0
# 0.31, 0.22, 0
# 0.33, 0.21, 0
# 0.35, 0.21, 0
# 0.37, 0.21, 0
# 0.39, 0.21, 0
# 0.41, 0.21, 0
# 0.43, 0.21, 0
# 0.45, 0.21, 0
# 0.47, 0.23, 0
# 0.49, 0.24, 0
# 0.51, 0.25, 0
# 0.53, 0.25, 0
# 0.55, 0.25, 0
# 0.57, 0.28, 0

# PERSISTENCE DIAGRAM
# 0.02, 0.19, 0
# 0.04, 0.20, 0
# 0.06, 0.19, 0
# 0.08, 0.19, 0
# 0.10, 0.19, 0
# 0.12, 0.19, 0
# 0.14, 0.19, 0
# 0.16, 0.19, 0
# 0.18, 0.19, 0
# 0.20, 0.19, 0
# 0.22, 0.19, 0
# 0.24, 0.18, 0
# 0.27, 0.18, 0
# 0.29, 0.18, 0
# 0.31, 0.18, 0
# 0.33, 0.17, 0
# 0.35, 0.17, 0
# 0.37, 0.17, 0
# 0.39, 0.17, 0
# 0.41, 0.17, 0
# 0.43, 0.17, 0
# 0.45, 0.17, 0
# 0.47, 0.19, 0
# 0.49, 0.19, 0
# 0.51, 0.19, 0
# 0.53, 0.19, 0
# 0.55, 0.19, 0
