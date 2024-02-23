import os
import sys

from collections import defaultdict
import glob
import numpy as np
import pandas as pd
import pickle
import pymol2
import scipy
from scipy.spatial.transform import Rotation
import shutil
import subprocess
import time
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from learning.SimpleUnet import SimpleHalfUnetModel
from learning.predict_coords import predict_coords
from utils.mrc_utils import MRCGrid
from paper.benchmark import PHENIX_DOCK_IN_MAP
from prepare_database.get_templates import REF_PATH_FV, REF_PATH_NANO, REF_PATH_FAB


def mwe():
    """Just sanity check for pymol procedures"""
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        pdb_em_path = "../data/pdb_em/"
        pdb = '7WP6'
        mrc = '32676'
        pdb_dir_path = os.path.join(pdb_em_path, f'{pdb}_{mrc}')
        pdb_path = os.path.join(pdb_dir_path, f'{pdb}.cif')
        p.cmd.load(pdb_path, 'in_pdb')
        selections = ['chain H or chain L', 'chain D or chain C', 'chain B or chain I']
        for i, selection in enumerate(selections):
            sel = f'in_pdb and ({selection})'
            p.cmd.extract(f"to_align", sel)
            coords = p.cmd.get_coords("to_align")
            print(pdb, selection)
            print(coords.shape)


def get_systems(csv_in='../data/csvs/sorted_filtered_test.csv',
                pdb_em_path="../data/pdb_em/",
                test_path="../data/testset/",
                nano=False):
    """
    The goal is to organize the test set in a clean repo with only necessary files
    :param csv_in:
    :param pdb_em_path:
    :param test_path:
    :param nano:
    :return:
    """
    # group mrc by pdb
    df = pd.read_csv(csv_in, index_col=0, dtype={'mrc': 'str'})
    os.makedirs(test_path, exist_ok=True)
    pdb_selections = defaultdict(list)
    df = df[['pdb', 'mrc', 'resolution', 'antibody_selection']]
    for i, row in df.iterrows():
        pdb, mrc, resolution, selection = row.values
        pdb_selections[(pdb.upper(), mrc, resolution)].append(selection)
    # Remove outlier systems with over 10 abs (actually just one)
    pdb_selections = {key: val for key, val in pdb_selections.items() if len(val) < 10}
    pickle.dump(pdb_selections, open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'wb'))
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        p.cmd.load(REF_PATH_FV, "ref_fv")
        p.cmd.load(REF_PATH_NANO, "ref_nano")
        p.cmd.load(REF_PATH_FAB, 'ref_fab')
        for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
            # if pdb != "8C7H":
            #     continue
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)}")
            pdb_dir_path = os.path.join(pdb_em_path, f'{pdb}_{mrc}')
            pdb_path = os.path.join(pdb_dir_path, f'{pdb}.cif')
            em_path = os.path.join(pdb_dir_path, "full_crop_resampled_2.mrc")

            # Make new dir
            new_dir_path = os.path.join(test_path, f'{pdb}_{mrc}')
            os.makedirs(new_dir_path, exist_ok=True)

            # Copy mrc file
            new_em_path = os.path.join(new_dir_path, "full_crop_resampled_2.mrc")
            new_pdb_path = os.path.join(new_dir_path, f"{pdb}.cif")
            shutil.copy(em_path, new_em_path)
            shutil.copy(pdb_path, new_pdb_path)

            # Get gt and rotated pdbs
            p.cmd.load(new_pdb_path, 'in_pdb')
            # for dockim comparison, we want to get more copies
            n_copies = 10 // (len(selections)) + 1
            for i, selection in enumerate(selections):
                outpath_gt = os.path.join(new_dir_path, f'gt_{"nano_" if nano else ""}{i}.pdb')
                sel = f'in_pdb and ({selection})'
                p.cmd.extract(f"to_align", sel)
                p.cmd.save(outpath_gt, "to_align")
                coords = p.cmd.get_coords("to_align")

                # To get COM consistence, we need to save the Fv part only
                if not nano:
                    residues_to_align = len(p.cmd.get_model("to_align").get_residues())
                    fab = residues_to_align > 300
                    # For fabs, first align the whole fab and then the Fv to its Fab,
                    # this drastically reduces the rmsd
                    if fab:
                        rmsd1 = p.cmd.align(mobile="ref_fab", target="to_align")[0]
                        rmsd2 = p.cmd.align(mobile="ref_fv", target="ref_fab")[0]
                        rmsd = rmsd1 + rmsd2
                        # if rmsd > 3:
                        #     print(pdb, rmsd1, rmsd2, fab, residues_to_align)
                    else:
                        rmsd = p.cmd.align(mobile="ref_fv", target="to_align")[0]
                        # if rmsd > 3:
                        #     print(pdb, rmsd, fab, residues_to_align)
                    outpath_gt_fv = os.path.join(new_dir_path, f'gt_fv_{i}.pdb')
                    p.cmd.save(outpath_gt_fv, "ref_fv")

                # To get COM consistence, we need to save the nano part only (for edge cases like megabodies)
                else:
                    rmsd = p.cmd.align(mobile="ref_nano", target="to_align")[0]
                    outpath_gt_nano = os.path.join(new_dir_path, f'gt_nano_{i}.pdb')
                    p.cmd.save(outpath_gt_nano, "ref_nano")

                # print(pdb, selection, n_copies, coords.shape)
                for k in range(n_copies):
                    rotated = Rotation.random().apply(coords)
                    translated = rotated + np.array([10, 20, 30])[None, :]
                    p.cmd.load_coords(translated, "to_align", state=1)
                    outpath_rotated = os.path.join(new_dir_path, f'rotated_{"nano_" if nano else ""}{i}_{k}.pdb')
                    p.cmd.save(outpath_rotated, 'to_align')
                p.cmd.delete("to_align")
            p.cmd.delete("in_pdb")


def make_predictions(nano=False, gpu=0, test_path="../data/testset/", use_mixed_model=True, sorted_split=True):
    """
    Now let's make predictions for this test set with ns_final model.
    :param nano:
    :param gpu:
    :param test_path:
    :return:
    """
    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))

    model = SimpleHalfUnetModel(classif_nano=use_mixed_model, num_feature_map=32)
    if use_mixed_model or nano:
        if sorted_split:
            model_path = os.path.join(script_dir, '../saved_models/ns_final_last.pth')
        else:
            model_path = os.path.join(script_dir, '../saved_models/nr_final_last.pth')
    else:
        if sorted_split:
            model_path = os.path.join(script_dir, '../saved_models/fs_final_last.pth')
        else:
            model_path = os.path.join(script_dir, '../saved_models/fr_final_last.pth')
    model.load_state_dict(torch.load(model_path))

    time_init = time.time()
    pred_number = {}
    with torch.no_grad():
        for step, (pdb, mrc, resolution) in enumerate(pdb_selections.keys()):
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)} in {time.time() - time_init}")
            pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')
            in_mrc = os.path.join(pdb_dir, "full_crop_resampled_2.mrc")
            if use_mixed_model or nano:
                out_name = os.path.join(pdb_dir, f'crai_pred{"_nano" if nano else ""}.pdb')
            else:
                out_name = os.path.join(pdb_dir, f'crai_pred_fab.pdb')
            predict_coords(mrc_path=in_mrc, outname=out_name, model=model, device=gpu, split_pred=True,
                           n_objects=10, thresh=0.2, classif_nano=use_mixed_model, use_pd=True, verbose=False)

            # TO GET num_pred
            transforms = predict_coords(mrc_path=in_mrc, outname=None, model=model, device=gpu, split_pred=True,
                                        n_objects=None, thresh=0.2, classif_nano=use_mixed_model, use_pd=True,
                                        verbose=False)
            pred_number[pdb] = len(transforms)
    if use_mixed_model:
        pickle_name = f'num_pred{"_nano" if nano else ""}.p'
    else:
        pickle_name = f'num_pred_fab.p'
    pickle.dump(pred_number, open(os.path.join(test_path, pickle_name), 'wb'))


def make_predictions_dockim(nano=False, test_path="../data/testset/"):
    """
    Now let's make predictions for this test set with ns_final model.
    :param nano:
    :param gpu:
    :param test_path:
    :return:
    """
    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))

    time_init = time.time()
    for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
        if not pdb == '7WP6':
            continue
        if not step % 20:
            print(f"Done {step} / {len(pdb_selections)} in {time.time() - time_init}")
        pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')
        in_mrc = os.path.join(pdb_dir, "full_crop_resampled_2.mrc")
        base_name = os.path.join(pdb_dir, f'rotated_{"nano_" if nano else ""}')
        input_dockim_to_dock = list(sorted(glob.glob(base_name + '*.pdb')))
        pdb_out = os.path.join(pdb_dir, f'dockim_pred{"_nano" if nano else ""}.pdb')
        try:
            t0 = time.time()
            # GET THE PDB TO DOCK
            if os.path.exists(pdb_out):
                return 5, "Already computed"
            # NOW WE CAN DOCK IN MAP
            cmd = f'{PHENIX_DOCK_IN_MAP} {" ".join(input_dockim_to_dock)} {in_mrc} pdb_out={pdb_out} resolution={resolution}'
            res = subprocess.run(cmd.split(), capture_output=True, timeout=5. * 3600)
            returncode = res.returncode
            if returncode > 0:
                return returncode, res.stderr.decode()

            # FINALLY WE NEED TO OFFSET THE RESULT BECAUSE OF CRAPPY PHENIX
            mrc_origin = MRCGrid.from_mrc(mrc_file=in_mrc).origin
            with pymol2.PyMOL() as p:
                p.cmd.load(pdb_out, 'docked')
                new_coords = p.cmd.get_coords('docked') + np.asarray(mrc_origin)[None, :]
                p.cmd.load_coords(new_coords, "docked", state=1)
                p.cmd.save(pdb_out, 'docked')
            time_tot = time.time() - t0
            return res.returncode, time_tot
        except TimeoutError as e:
            return 1, e
        except Exception as e:
            print(e)
            return 2, e


def compute_matching_hungarian(actual_pos, pred_pos, thresh=10):
    dist_matrix = scipy.spatial.distance.cdist(pred_pos, actual_pos)
    # print(actual_pos, pred_pos)
    gt_found = []
    for i in range(1, len(pred_pos) + 1):
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix[:i])
        position_dists = dist_matrix[row_ind, col_ind]
        found = np.sum(position_dists < thresh)
        gt_found.append(found)
    # print(gt_found)
    # print(position_dists)
    # print()
    return gt_found


def get_hit_rates(nano=False, test_path="../data/testset/", use_mixed_model=True):
    """
    Go over the predictions and computes the hit rates with each number of systems.
    :param nano:
    :param test_path:
    :return:
    """
    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))

    time_init = time.time()
    all_res = {}
    with pymol2.PyMOL() as p:
        for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
            # if pdb not in ['7U0X', '7XDB', '7XJ6', '7YVI', '7YVN', '7YVO', '7ZJL', '8DWW', '8DWX', '8DWY',
            #                    '8GTP', '8H07', '8HEC', '8IL3']:
            #     continue
            # print(pdb)
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)} in {time.time() - time_init}")

            pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')

            # First get the (sorted) list of predicted com
            predicted_com = []
            for i in range(10):
                # for i in range(len(selections)):
                if use_mixed_model or nano:
                    out_name = os.path.join(pdb_dir, f'crai_pred{"_nano" if nano else ""}_{i}.pdb')
                else:
                    out_name = os.path.join(pdb_dir, f'crai_pred_fab{"_nano" if nano else ""}_{i}.pdb')
                if not os.path.exists(out_name):
                    # Not sure why but sometimes fail to produce 10 systems.
                    # Still gets 5-6 for small systems. Maybe the grid is too small.
                    # print(out_name)
                    predicted_com.append((0, 0, 0))
                    continue
                p.cmd.load(out_name, 'crai_pred')
                predictions = p.cmd.get_coords(f'crai_pred')
                com = np.mean(predictions, axis=0)
                predicted_com.append(com)
                p.cmd.delete('crai_pred')

            # Now get the list of GT com
            gt_com = []
            for i in range(len(selections)):
                # We use the Fv GT in the vase of Fabs
                gt_name = os.path.join(pdb_dir, f'gt_{"nano_" if nano else "fv_"}{i}.pdb')
                p.cmd.load(gt_name, 'gt')
                gt_coords = p.cmd.get_coords('gt')
                com = np.mean(gt_coords, axis=0)
                gt_com.append(com)
                p.cmd.delete('gt')

            hits_thresh = compute_matching_hungarian(gt_com, predicted_com)
            gt_hits_thresh = list(range(1, len(gt_com) + 1)) + [len(gt_com)] * (len(predicted_com) - len(gt_com))
            all_res[pdb] = (gt_hits_thresh, hits_thresh, resolution)
    if use_mixed_model or nano:
        outname = os.path.join(test_path, f'all_res{"_nano" if nano else ""}.p')
    else:
        outname = os.path.join(test_path, f'all_res_fab.p')
    pickle.dump(all_res, open(outname, 'wb'))


def string_rep(sorted_split=None, nano=None, mixed=None, num=None):
    s = ""
    if sorted_split is not None:
        s += 'Sorted ' if sorted_split else 'Random '
    if nano is not None:
        s += 'Nano ' if nano else 'Fab '
    if mixed is not None:
        s += 'Mixed ' if mixed else 'NonMixed '
    if num is not None:
        s += 'Num' if num else 'Thresh '
    return s


if __name__ == '__main__':
    # TODO : Get a two step RMSD for fabs to avoid misalignments
    # TODO : dockim
    # TODO : understand why n<10 sometimes
    # mwe()

    # GET DATA
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            csv_in = f'../data/{"nano_" if nano else ""}csvs/{"sorted_" if sorted_split else ""}filtered_test.csv'
            print('Getting data for ', string_rep(sorted_split=sorted_split, nano=nano))
            get_systems(csv_in=csv_in, nano=nano, test_path=test_path)
            # Now let us get the prediction in all cases
            for mixed in [False, True]:
                # No models are dedicated to nano only
                if nano and not mixed:
                    continue
                print('Making predictions for :', string_rep(nano=nano, mixed=mixed))
                make_predictions(nano=nano, test_path=test_path, use_mixed_model=mixed, gpu=1)
                # make_predictions_dockim(nano=nano)
                get_hit_rates(nano=nano, test_path=test_path, use_mixed_model=mixed)
