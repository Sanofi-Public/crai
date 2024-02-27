import os
import sys

import functools
import multiprocessing
import numpy as np
import pickle
import pymol2
import string
from scipy.spatial.transform import Rotation
import subprocess
import time
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from paper.predict_test import get_pdbsels, string_rep, compute_matching_hungarian
from utils.mrc_utils import MRCGrid
from utils.python_utils import init

PHENIX_DOCK_IN_MAP = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.dock_in_map"
UPPERCASE = string.ascii_uppercase
LOWERCASE = string.ascii_lowercase


def get_systems(csv_in='../data/csvs/sorted_filtered_test.csv',
                test_path="../data/testset/",
                nano=False):
    os.makedirs(test_path, exist_ok=True)
    out_name_pdbsels = os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p')
    pdb_selections = get_pdbsels(csv_in=csv_in, out_name=out_name_pdbsels)
    rotated_chain_mapping = {}
    # Get rotated pdbs
    # for dockim comparison, we want to get more copies and also to rename the chains.
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        all_pdb_chain_mapping = {}
        for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)}")
            new_dir_path = os.path.join(test_path, f'{pdb}_{mrc}')
            new_pdb_path = os.path.join(new_dir_path, f"{pdb}.cif")
            p.cmd.load(new_pdb_path, 'in_pdb')
            # Using 10%1 avoids choosing an arbitrary number like 3 copies of A and B and 2 of C and D (10=3+3+2+2)
            # => we get 3 copies of each
            # We cannot afford n_copies*len(selections) to go over 13 for fabs (since we only have 26 uppercase)
            # 5x2, 4x3, 3x4, 4x3, 2x5, 2x6 works but fails for more systems (we don't have the issue)
            n_copies = 10 // (len(selections)) + (10 % len(selections) != 0)
            # print(n_copies, n_copies * len(selections))
            last_chain = 0
            pdb_chain_mapping = {}
            for i, selection in enumerate(selections):
                sel = f'in_pdb and ({selection})'
                p.cmd.extract(f"to_align", sel)
                original_chains = selection.split(' or ')
                original_chains = [chain.strip('chain ') for chain in original_chains]
                current_chains = original_chains
                coords = p.cmd.get_coords("to_align")
                # print(pdb, selection, n_copies, coords.shape)
                # Make rotated copies of abs
                for k in range(n_copies):
                    rotated = Rotation.random().apply(coords)
                    translated = rotated + np.array([10, 20, 30])[None, :]
                    p.cmd.load_coords(translated, "to_align", state=1)

                    # Handle chain renaming that is important for dockim output
                    # nAbs are not a big problem, but we need to avoid collision for Fabs.
                    # Save in a dict map = {(i,k): (old_chains, new_chains)}
                    if nano:
                        new_chain = LOWERCASE[last_chain]
                        last_chain += 1
                        p.cmd.alter('to_align', f"chain='{new_chain}'")
                        pdb_chain_mapping[(i, k)] = (original_chains, new_chain)
                    else:
                        # We want to go from X,Y => A,B avoiding collisions
                        old_chain_1, old_chain_2 = current_chains
                        new_chain_1, new_chain_2, temp_chain = [UPPERCASE[last_chain + offset] for offset in range(3)]
                        last_chain += 2
                        # Y != A: X,Y -> A,Y -> A,B
                        if not new_chain_1 == old_chain_2:
                            p.cmd.alter(f'to_align and chain {old_chain_1}', f"chain='{new_chain_1}'")
                            p.cmd.alter(f'to_align and chain {old_chain_2}', f"chain='{new_chain_2}'")
                        # We now want X,A => A,B
                        # X,A -> C,A -> C,B -> A,B
                        else:
                            p.cmd.alter(f'to_align and chain {old_chain_1}', f"chain='{temp_chain}'")
                            p.cmd.alter(f'to_align and chain {old_chain_2}', f"chain='{new_chain_2}'")
                            p.cmd.alter(f'to_align and chain {temp_chain}', f"chain='{new_chain_1}'")
                        current_chains = new_chain_1, new_chain_2
                        pdb_chain_mapping[(i, k)] = ((old_chain_1, old_chain_2), (new_chain_1, new_chain_2))
                    outpath_rotated = os.path.join(new_dir_path, f'rotated_{"nano_" if nano else ""}{i}_{k}.pdb')
                    p.cmd.save(outpath_rotated, 'to_align')
                p.cmd.delete("to_align")
            # print(pdb, pdb_chain_mapping)
            all_pdb_chain_mapping[pdb] = pdb_chain_mapping
            p.cmd.delete("in_pdb")
    outname_mapping = os.path.join(test_path, f'dockim_chain_map{"_nano" if nano else ""}.p')
    pickle.dump(all_pdb_chain_mapping, open(outname_mapping, 'wb'))


def dock_one(pdb_paths, mrc_path, resolution, out_path, recompute=False):
    """
    Run dock in map on input_pdbs.
    The mrc file is supposed to be a custom one in mrc format.
    Thus, we open it and get its origin as it is not correctly read by phenix.
    Then we feed dock_in_map with the right number of fabs and fvs and use dock in map to store them.
    """

    mrc_path = os.path.abspath(mrc_path)
    pdb_paths = [os.path.abspath(pdb_path) for pdb_path in pdb_paths]
    _, mrc_extension = os.path.splitext(mrc_path)
    assert mrc_extension == '.mrc'

    try:
        t0 = time.time()
        # GET THE PDB TO DOCK
        if os.path.exists(out_path) and not recompute:
            return 5, "Already computed"

        # NOW WE CAN DOCK IN MAP
        cmd = f'{PHENIX_DOCK_IN_MAP} {" ".join(pdb_paths)} {mrc_path} pdb_out={out_path} resolution={resolution}'
        # print(len(pdb_paths), os.path.basename(pdb_paths[-1]), cmd)
        # return 0, 1
        res = subprocess.run(cmd.split(), capture_output=True, timeout=5. * 3600)
        returncode = res.returncode
        if returncode > 0:
            return returncode, res.stderr.decode()

        # FINALLY WE NEED TO OFFSET THE RESULT BECAUSE OF CRAPPY PHENIX
        mrc_origin = MRCGrid.from_mrc(mrc_file=mrc_path).origin
        with pymol2.PyMOL() as p:
            p.cmd.load(out_path, 'docked')
            new_coords = p.cmd.get_coords('docked') + np.asarray(mrc_origin)[None, :]
            p.cmd.load_coords(new_coords, "docked", state=1)
            p.cmd.save(out_path, 'docked')
        time_tot = time.time() - t0
        return res.returncode, time_tot
    except TimeoutError as e:
        return 1, e
    except Exception as e:
        return 2, e


def make_predictions_dockim(nano=False, test_path="../data/testset/"):
    """
    Make predictions with all combinations of rotated units.
    :param nano:
    :param test_path:
    :return:
    """
    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))
    outname_mapping = os.path.join(test_path, f'dockim_chain_map{"_nano" if nano else ""}.p')
    all_pdb_chain_mapping = pickle.load(open(outname_mapping, 'rb'))

    # Prepare input list
    dockim_inputs = list()
    for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
        pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')
        in_mrc = os.path.join(pdb_dir, "full_crop_resampled_2.mrc")
        input_pdbs = list()
        # Start with 0,0 then 1,0 then 0,1 and so on to have all unit at n_pred=num_gt
        for n_pred, (i, k) in enumerate(sorted(all_pdb_chain_mapping[pdb].keys(), key=lambda x: x[1])):
            # Maximum 10 predictions
            if n_pred >= 10:
                continue
            new_input = os.path.join(pdb_dir, f'rotated_{"nano_" if nano else ""}{i}_{k}.pdb')
            out_path = os.path.join(pdb_dir, f'dockim_pred{"_nano" if nano else ""}_{n_pred}.pdb')
            input_pdbs.append(new_input)
            dockim_inputs.append((tuple(input_pdbs), in_mrc, resolution, out_path))

    # Parallel computation
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=40, initializer=init, initargs=(l,), )
    dock = functools.partial(dock_one, recompute=False)
    results = pool.starmap(dock, tqdm(dockim_inputs, total=len(dockim_inputs)))

    # Parse results
    all_results = []
    all_errors = []
    for i, (return_code, runtime) in enumerate(results):
        if return_code == 0:
            all_results.append(runtime)
        else:
            all_results.append(-return_code)
            all_errors.append((return_code, runtime))
    for x in all_errors:
        print(x)
    pickle.dump((all_results, all_errors), open('results_timing_dockim.p', 'wb'))


def get_hit_rates_dockim(nano=False, test_path="../data/testset/"):
    """
    Go over the predictions and computes the hit rates with each number of systems.
    :param nano:
    :param test_path:
    :return:
    """
    pdb_selections = pickle.load(open(os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p'), 'rb'))
    outname_mapping = os.path.join(test_path, f'dockim_chain_map{"_nano" if nano else ""}.p')
    all_pdb_chain_mapping = pickle.load(open(outname_mapping, 'rb'))

    time_init = time.time()
    all_res = {}
    with pymol2.PyMOL() as p:
        for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)} in {time.time() - time_init}")

            pdb_dir = os.path.join(test_path, f'{pdb}_{mrc}')

            # Get the list of GT com
            gt_com = []
            for i in range(len(selections)):
                # We use the Fv GT in the vase of Fabs
                gt_name = os.path.join(pdb_dir, f'gt_{"nano_" if nano else "fv_"}{i}.pdb')
                p.cmd.load(gt_name, 'gt')
                gt_coords = p.cmd.get_coords('gt')
                com = np.mean(gt_coords, axis=0)
                gt_com.append(com)
                p.cmd.delete('gt')
            max_com = np.max(np.stack(gt_com), axis=0)

            # With dockim, we cannot sort, so we have to compute the hits separately for all 10 predictions
            hits_thresh = []
            for i in range(10):
                out_name = os.path.join(pdb_dir, f'dockim_pred{"_nano" if nano else ""}_{i}.pdb')
                if not os.path.exists(out_name):
                    hits_thresh.append(0)
                    continue
                else:
                    if nano:
                        pymol_chain_sels = [f"chain {LOWERCASE[j]}" for j in range(i + 1)]
                    else:
                        pymol_chain_sels = [f"chain {UPPERCASE[2 * j]} or chain {UPPERCASE[2 * j + 1]}"
                                            for j in range(i + 1)]
                    pymol_name = 'dockim_pred_i'
                    p.cmd.load(out_name, pymol_name)
                    predicted_com = []
                    for pymol_sel in pymol_chain_sels:
                        predictions = p.cmd.get_coords(f'{pymol_name} and ({pymol_sel})')
                        if predictions is None:
                            continue
                        com = np.mean(predictions, axis=0)
                        predicted_com.append(com)
                    p.cmd.delete(pymol_name)
                    hits_thresh_i = compute_matching_hungarian(gt_com, predicted_com)[-1]
                    hits_thresh.append(hits_thresh_i)
            gt_hits_thresh = list(range(1, len(gt_com) + 1)) + [len(gt_com)] * (10 - len(gt_com))
            all_res[pdb] = (gt_hits_thresh, hits_thresh, resolution)
    outname = os.path.join(test_path, f'all_res_dockim{"_nano" if nano else ""}.p')
    pickle.dump(all_res, open(outname, 'wb'))


if __name__ == '__main__':
    # GET DATA
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            # for nano in [True]:
            csv_in = f'../data/{"nano_" if nano else ""}csvs/{"sorted_" if sorted_split else ""}filtered_test.csv'
            print('Getting data for ', string_rep(sorted_split=sorted_split, nano=nano))
            # get_systems(csv_in=csv_in, nano=nano, test_path=test_path)
            # Now let us get the prediction in all cases

            print('Making predictions for :', string_rep(nano=nano))
            # make_predictions_dockim(nano=nano, test_path=test_path)

            print('Getting hit rates for :', string_rep(nano=nano))
            get_hit_rates_dockim(nano=nano, test_path=test_path)
