import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from paper.predict_test import string_rep


def compute_hr(nano=False, test_path='../data/testset', num_setting=False, use_mixed_model=True):
    """
    Compute the HR metric in the sense of the paper (using the actual number of prediction)
    :param nano:
    :param test_path:
    :return:
    """
    if use_mixed_model or nano:
        all_res_path = os.path.join(test_path, f'all_res{"_nano" if nano else ""}.p')
        num_pred_path = os.path.join(test_path, f'num_pred{"_nano" if nano else ""}.p')
    else:
        all_res_path = os.path.join(test_path, f'all_res_fab.p')
        num_pred_path = os.path.join(test_path, f'num_pred_fab.p')

    all_res = pickle.load(open(all_res_path, 'rb'))
    num_pred_all = pickle.load(open(num_pred_path, 'rb'))
    all_hr = {}
    overpreds_list = []
    underpreds_list = []
    for pdb, (gt_hits_thresh, hits_thresh, resolution) in sorted(all_res.items()):
        # Systems containing both Fab and nAb in random split
        if test_path == '../data/testset_random':
            if pdb in ['7PIJ', '7SK5', '7WPD', '7XOD', '7ZLJ', '8HIK']:
                continue

        if test_path == '../data/testset':
            # Misclassified nano in sorted
            if pdb == '7YC5':
                continue


        gt_hits_thresh = np.array(gt_hits_thresh)
        hits_thresh = np.array(hits_thresh)
        num_gt = np.max(gt_hits_thresh)
        if num_setting:
            num_pred = num_gt
        else:
            num_pred = num_pred_all[pdb]
        overpreds = max(0, num_pred - num_gt)
        found_hits = hits_thresh[num_pred - 1]
        underpreds = num_gt - found_hits
        errors = overpreds + underpreds

        # PRINT
        if overpreds > 0:
            overpreds_list.append((pdb, overpreds))
            # Was this overpred useful ? (if found_hits > hits_thresh[num_gt - 1])
            # Actually useful only twice for fabs and twice for nano
            useful = found_hits > hits_thresh[num_gt - 1]
            # print(pdb, num_pred, num_gt, found_hits, hits_thresh[num_gt - 1], hits_thresh, useful)
        if underpreds > 0:
            # Would we find it with more hits ?
            # Not so much with Fabs, some are close but further than 10, others are just missed.
            # 100% yes with nano

            # more_would_help = hits_thresh[-1] > found_hits
            # print(pdb, num_pred, num_gt, found_hits, hits_thresh[num_gt - 1], hits_thresh, more_would_help)
            underpreds_list.append((pdb, underpreds))
        if overpreds > 0 and underpreds > 0:
            print(pdb, 'winner !')
        all_hr[pdb] = (errors, num_gt)
        # if errors > 0:
        #     print(f'For PDB {pdb}, {"overprediction" if overpreds > 0 else "underprediction"} :'
        #           f'predicted num: {num_pred}, GT num {num_gt}, all hits{hits_thresh}')
    print('Overpredictions : ', sum([x[1] for x in overpreds_list]), len(overpreds_list), overpreds_list)
    print('Underpredictions : ', sum([x[1] for x in underpreds_list]), len(underpreds_list), underpreds_list)

    hit_rate_sys = np.mean([100 * (1 - errors / num_gt) for errors, num_gt in all_hr.values()])
    hit_rate_ab = 100 * (1 - np.sum([errors for errors, _ in all_hr.values()]) / np.sum(
        [num_gt for _, num_gt in all_hr.values()]))
    # print('sys : ',hit_rate_sys)
    # print('ab : ', hit_rate_ab)
    print(f"{hit_rate_sys:.1f}")
    print(f"{hit_rate_ab:.1f}")


def plot_pr_curve(nano=False, test_path='../data/testset', use_mixed_model=True):
    if use_mixed_model or nano:
        outname = os.path.join(test_path, f'all_res{"_nano" if nano else ""}.p')
    else:
        outname = os.path.join(test_path, f'all_res_fab.p')
    all_res = pickle.load(open(outname, 'rb'))
    all_precisions = []
    all_gt = []
    all_preds = []
    for pdb, (gt_hits_thresh, hits_thresh, resolution) in all_res.items():
        if pdb == '7YC5':
            continue
        gt_hits_thresh = np.array(gt_hits_thresh)
        hits_thresh = np.array(hits_thresh)
        num_gt = np.max(gt_hits_thresh)
        precision = hits_thresh / gt_hits_thresh
        all_precisions.append(precision)
        all_gt.append(gt_hits_thresh / num_gt)
        all_preds.append(hits_thresh / num_gt)
        if precision[-1] < 0.9:
            print(pdb, num_gt, hits_thresh)
    all_precisions = np.stack(all_precisions)
    all_precisions_mean = np.mean(all_precisions, axis=0)
    all_precisions_std = np.std(all_precisions, axis=0)

    all_gt = np.stack(all_gt)
    all_gt_mean = np.mean(all_gt, axis=0)
    all_gt_std = np.std(all_gt, axis=0)

    all_preds = np.stack(all_preds)
    all_preds_mean = np.mean(all_preds, axis=0)
    all_preds_std = np.std(all_preds, axis=0)

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['text.usetex'] = True
    plt.rc('grid', color='grey', alpha=0.5)
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel(r'Num prediction')
    ax.set_ylabel(r'Hits')

    x = range(1, 11)
    ax = plt.gca()

    def plot_in_between(ax, mean, std, **kwargs):
        ax.plot(x, mean, **kwargs)
        ax.fill_between(x, mean + std, mean - std, alpha=0.5)

    # plot_in_between(ax, all_precisions_mean, all_precisions_std, label=rf'Fractional precision{"_fab" if fab else ""}')
    plot_in_between(ax, all_gt_mean, all_gt_std, label=r'\texttt{Ground Truth}')
    plot_in_between(ax, all_preds_mean, all_preds_std, label=r'\texttt{CrAI}')

    plt.xlim((1, 10))
    plt.ylim((0.5, 1))
    plt.legend()
    # plt.show()


if __name__ == '__main__':
    # plot_pr_curve(nano=False, use_mixed_model=True)
    # plot_pr_curve(nano=False)
    # plt.show()
    # plot_pr_curve(nano=True)
    # plt.show()

    # Compute the # systems with both Fabs and nanobodies as they bug the validation a bit
    # for sorted_split in [True, False]:
    #     test_path = f'../data/testset{"" if sorted_split else "_random"}'
    #     systems = []
    #     for nano in [True, False]:
    #         pdbsel = os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p')
    #         systems.append(pickle.load(open(pdbsel, 'rb')))
    #     nab_pdb = set([pdb for pdb, _, _ in systems[0].keys()])
    #     fab_pdb = set([pdb for pdb, _, _ in systems[1].keys()])
    #     print('Num in nAbs:', len(nab_pdb),
    #           'Num in Fabs:', len(fab_pdb),
    #           'Num in both:', len(nab_pdb.intersection(fab_pdb)))
    #     print(sorted(nab_pdb.intersection(fab_pdb)))

    for sorted_split in [True]:
    # for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False]:
        # for nano in [False, True]:
            # for mixed in [False, True]:
            for mixed in [True]:
                if nano and not mixed:
                    continue
                # for num_setting in [False]:
                for num_setting in [True, False]:
                    print('Results HR for :', string_rep(sorted_split=sorted_split,
                                                         nano=nano,
                                                         mixed=mixed,
                                                         num=num_setting))
                    compute_hr(test_path=test_path, nano=nano, use_mixed_model=mixed, num_setting=num_setting)
