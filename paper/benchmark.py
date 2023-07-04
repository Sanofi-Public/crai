import os
import pickle
import sys

import functools
import multiprocessing
import numpy as np
import pandas as pd
import pymol2
import scipy
from scipy.spatial.transform import Rotation as R
import string
import subprocess
import time
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils.mrc_utils import MRCGrid
from prepare_database.get_templates import REF_PATH_FV, REF_PATH_FAB
from prepare_database.process_data import get_pdb_selection
from utils.python_utils import init
from utils.object_detection import pdbsel_to_transform

PHENIX_DOCK_IN_MAP = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.dock_in_map"
UPPERCASE = string.ascii_uppercase
LOWERCASE = string.ascii_lowercase


def copy_templates():
    """
    Stupid idea, the goal is to produce 11 (max number of antibodies in the data) copies of the reference files
     Then we rename the 11 pair of chains with a number, so that alignments produced by dock_in_map can be more
     easily processed.
    fv1 => Aa, fv2=> Bb,...fv11=>Ll, fab1=>Mm,...fab11=>Ww
    """

    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        p.cmd.load(REF_PATH_FV, 'ref_fv')
        fv_file_path, _ = os.path.splitext(REF_PATH_FV)
        p.cmd.load(REF_PATH_FAB, 'ref_fab')
        fab_file_path, _ = os.path.splitext(REF_PATH_FAB)
        for i in range(11):
            fv_save_path = f"{fv_file_path}_{i + 1}.pdb"
            fv_sel_i = f"fv_{i + 1}"
            p.cmd.copy(fv_sel_i, "ref_fv")
            p.cmd.alter(f'{fv_sel_i} and chain H', f"chain='{UPPERCASE[i]}'")
            p.cmd.alter(f'{fv_sel_i} and chain L', f"chain='{LOWERCASE[i]}'")
            p.cmd.save(fv_save_path, fv_sel_i)

            # Fab chains are offset to avoid collisions
            fab_save_path = f"{fab_file_path}_{i + 1}.pdb"
            fab_sel_i = f"fab_{i + 1}"
            p.cmd.copy(fab_sel_i, "ref_fab")
            p.cmd.alter(f'{fab_sel_i} and chain H', f"chain='{UPPERCASE[i + 11]}'")
            p.cmd.alter(f'{fab_sel_i} and chain L', f"chain='{LOWERCASE[i + 11]}'")
            p.cmd.save(fab_save_path, fab_sel_i)


def get_num_fabs_fvs(pdb_path, selections):
    fabs, fvs = 0, 0
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        p.cmd.load(pdb_path, 'in_pdb')
        for selection in selections:
            sel = f'in_pdb and ({selection})'
            p.cmd.extract("to_align", sel)
            residues_to_align = len(p.cmd.get_model("to_align").get_residues())
            if residues_to_align < 300:
                fvs += 1
            else:
                fabs += 1
    return fabs, fvs


def dock_chains(mrc_path, pdb_path, selections, resolution=4., use_template=False):
    """
    Run dock in map on one chain.
    The mrc file is supposed to be a custom one in mrc format.
    Thus, we open it and get its origin as it is not correctly read by phenix.
    Then we feed dock_in_map with the right number of fabs and fvs and use dock in map to store them.
    """
    # with correct pdb <1min
    # With rotated (out of box) : 24 minutes 30
    # With centered : 27 min 7
    # With centered + centered_2 : 41 min 38
    # Additional test here, the two chains are named in the same way, how is phenix going to deal with that
    # Phenix follows the name for the first chain, then rename the other models' names following alphabetical order.
    # With centered + centered_2 + resampled_map: 16 min 35 but does not find correct solution...
    # With centered + centered_2 + carved_map : Same result, but actually the 'error' is just the origin offset !
    # With reference + resampled_map : Last test : can we just use an unaligned ref
    #     with the resampled and get the offset ? => yes in 13 minutes !

    mrc_path = os.path.abspath(mrc_path)
    pdb_path = os.path.abspath(pdb_path)
    _, mrc_extension = os.path.splitext(mrc_path)
    assert mrc_extension == '.mrc'
    try:
        t0 = time.time()
        # GET THE PDB TO DOCK (for now copies of the reference)
        dir_path = os.path.dirname(pdb_path)
        pdb_file = os.path.basename(pdb_path)
        pdb, _ = os.path.splitext(pdb_file)
        selections = [res[0] for res in selections]
        if use_template:
            fabs, fvs = get_num_fabs_fvs(pdb_path)
            fv_file_path, _ = os.path.splitext(REF_PATH_FV)
            fab_file_path, _ = os.path.splitext(REF_PATH_FAB)
            to_dock = [f"{fv_file_path}_{i + 1}.pdb" for i in range(fvs)] + \
                      [f"{fab_file_path}_{i + 1}.pdb" for i in range(fabs)]
            pdb_out = os.path.join(dir_path, 'output_dock_in_map.pdb')

        # We could also implement extraction of the actual chains (is that cheating ?)
        else:
            to_dock = []
            with pymol2.PyMOL() as p:
                p.cmd.feedback("disable", "all", "everything")
                p.cmd.load(pdb_path, 'in_pdb')
                for i, selection in enumerate(selections):
                    sel = f'in_pdb and ({selection})'
                    p.cmd.extract("to_align", sel)
                    coords = p.cmd.get_coords("to_align")
                    rotated = R.random().apply(coords)
                    translated = rotated + np.array([10, 20, 30])[None, :]
                    p.cmd.load_coords(translated, "to_align", state=1)
                    outname = os.path.join(os.path.dirname(pdb_path), f'to_dock_{i}.pdb')
                    p.cmd.save(outname, 'to_align')
                    to_dock.append(outname)
            pdb_out = os.path.join(dir_path, 'output_dock_in_map_actual.pdb')

        # NOW WE CAN DOCK IN MAP

        cmd = f'{PHENIX_DOCK_IN_MAP} {" ".join(to_dock)} {mrc_path} pdb_out={pdb_out} resolution={resolution}'
        res = subprocess.run(cmd.split(), capture_output=True, timeout=5. * 3600)
        returncode = res.returncode
        if returncode > 0:
            return returncode, res.stderr.decode()

        # FINALLY WE NEED TO OFFSET THE RESULT BECAUSE OF CRAPPY PHENIX
        mrc_origin = MRCGrid.from_mrc(mrc_file=mrc_path).origin
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
        return 2, e


def compute_all_dockinmap(csv_in, csv_out, datadir_name='../data/pdb_em', use_template=False):
    # Prepare input list
    all_systems = "../data/csvs/filtered.csv"
    pdb_selections = get_pdb_selection(csv_in=all_systems, columns=['antibody_selection'])
    columns = ["pdb", "mrc", "resolution"]
    df = pd.read_csv(csv_in, index_col=0, dtype={'mrc': 'str'})[columns]
    df = df.groupby("pdb", as_index=False).nth(0).reset_index(drop=True)
    to_process = []
    for i, row in df.iterrows():
        pdb, mrc, resolution = row.values
        pdb = pdb.upper()
        dirname = f"{pdb}_{mrc}"
        system_dir = os.path.join(datadir_name, dirname)
        pdb_path = os.path.join(system_dir, f"{pdb}.cif")
        mrc_path = os.path.join(system_dir, "full_crop_resampled_2.mrc")
        selections = pdb_selections[pdb]
        to_process.append((mrc_path, pdb_path, selections, resolution))

    # Parallel computation
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=32, initializer=init, initargs=(l,), )
    dock = functools.partial(dock_chains, use_template=use_template)
    results = pool.starmap(dock, tqdm(to_process, total=len(to_process)))

    # Parse results
    all_results = []
    all_errors = []
    for i, (return_code, runtime) in enumerate(results):
        if return_code == 0:
            all_results.append(runtime)
        else:
            all_results.append(-return_code)
            all_errors.append((return_code, runtime))
    df['dock_runtime'] = all_results
    df.to_csv(csv_out)
    for x in all_errors:
        print(x)


def parse_one(outfile, gt_pdb, selections, use_template=False):
    try:
        selections = [x[0] for x in selections]
        gt_transforms = pdbsel_to_transform(gt_pdb, selections)
        if use_template:
            fabs, fvs = get_num_fabs_fvs(gt_pdb, selections)
            fvs_selections = [f"chain '{UPPERCASE[i]} or chain {LOWERCASE[i]}" for i in range(fvs)]
            fabs_selections = [f"chain '{UPPERCASE[i + 11]} or chain {LOWERCASE[i + 11]}" for i in range(fabs)]
            selections = fvs_selections + fabs_selections
        predicted_transforms = pdbsel_to_transform(outfile, selections, cache=False)
        pred_translations = [res[1] for res in predicted_transforms]
        gt_translations = [res[1] for res in gt_transforms]
        dist_matrix = scipy.spatial.distance.cdist(pred_translations, gt_translations)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix)
        position_dists = dist_matrix[row_ind, col_ind]
        mean_dist = float(position_dists.mean())
        return mean_dist, position_dists, col_ind
    except Exception as e:
        print(e)
        return None


def parse_all_dockinmap(csv_in, parsed_out, pdb_selections, use_template=False):
    df_raw = pd.read_csv(csv_in, index_col=0, dtype={'mrc': 'str'})
    all_res = dict()
    for i, row in df_raw.iterrows():
        pdb, mrc, resolution, dock_runtime = row.values
        datadir_name = "../data/pdb_em"
        dirname = f"{pdb.upper()}_{mrc}"
        pdb_path = os.path.join(datadir_name, dirname, f"{pdb.upper()}.cif")
        out_name = "output_dock_in_map.pdb" if use_template else "output_dock_in_map_actual.pdb"
        out_path = os.path.join(datadir_name, dirname, out_name)
        selections = pdb_selections[pdb.upper()]
        res = parse_one(out_path, pdb_path, selections, use_template=False)
        all_res[pdb] = res
        if not i % 20:
            print(f"Done {i}/{len(df_raw)}")
    pickle.dump(all_res, open(parsed_out, 'wb'))


if __name__ == '__main__':
    pass
    # test templates
    # copy_templates()

    # test one
    # datadir_name = "../data/pdb_em"
    # dirname = "6V4N_21042"
    # pdb_name, mrc_name = dirname.split("_")
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
    # resampled_path = os.path.join(datadir_name, dirname, "full_crop_resampled_2.mrc")
    # all_systems = "../data/csvs/filtered.csv"
    # pdb_selections = get_pdb_selection(csv_in=all_systems, columns=['antibody_selection'])
    # selections = pdb_selections[pdb_name.upper()]
    # res = dock_chains(pdb_path=pdb_path, mrc_path=resampled_path, selections=selections)
    # print(res)

    # test all
    # csv_in = '../data/csvs/filtered.csv'
    # use_template = True
    # csv_out = f'../data/csvs/benchmark{"_actual" if not use_template else ""}.csv'
    # compute_all_dockinmap(csv_in=csv_in, csv_out=csv_out, use_template=use_template)

    # NO TEMPLATE
    # (1, 'Sorry: Unknown charge:\n  "ATOM    140  CA  ALA K 140 .*. I    C "\n                                       ^^\n')
    # (1, 'Sorry: No solution found...you might try with quick=False\n')
    # (1, 'Sorry: No solution found...you might try with quick=False\n')
    # (2, CmdException('failed to open file "/home/mallet/projects/crIA-EM/data/pdb_em/7TEQ_25849/output_dock_in_map_actual.pdb"'))
    # (1, 'Sorry: Unknown charge:\n  "ATOM   1541  N   LYS H 212 .*. D    N "\n                                       ^^\n')
    # (1, 'Sorry: Unknown charge:\n  "ATOM    916  N   ALA H 125 .*. C    N "\n                                       ^^\n')
    # (2, CmdException('failed to open file "/home/mallet/projects/crIA-EM/data/pdb_em/7OH1_12891/output_dock_in_map_actual.pdb"'))
    # (1, 'Sorry: No solution found...you might try with quick=False\n')
    # (1, 'Sorry: No solution found...you might try with quick=False\n')
    # (2, TimeoutExpired(['/users/mallet/bin/phenix-1.20.1-4487/build/bin/phenix.dock_in_map', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/to_dock_0.pdb', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/to_dock_1.pdb', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/to_dock_2.pdb', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/to_dock_3.pdb', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/to_dock_4.pdb', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/to_dock_5.pdb', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/to_dock_6.pdb', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/to_dock_7.pdb', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/to_dock_8.pdb', '/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/full_crop_resampled_2.mrc', 'pdb_out=/home/mallet/projects/crIA-EM/data/pdb_em/7E8C_31014/output_dock_in_map_actual.pdb', 'resolution=3.16'],
    # 18000.0))
    # (1, 'Sorry: No solution found...you might try with quick=False\n')
    # (1, 'Sorry: Error in SS definitions, most likely atoms are absent for one of them.\n')
    # (1, 'Sorry: Unknown charge:\n  "ATOM   1298  N   ASN F 176 .*. E    N "\n                                       ^^\n')
    # (1, 'Sorry: Unknown charge:\n  "ATOM   1141  N   ASP L 152 .*. C    N "\n                                       ^^\n')
    # (1, 'Sorry: No solution found...you might try with quick=False\n')
    # (1, 'Sorry: Unknown charge:\n  "ATOM   2795  N   SER L 158 .*. F    N "\n                                       ^^\n')
    # (1, 'Sorry: Unknown charge:\n  "ATOM   1541  N   LYS H 212 .*. D    N "\n                                       ^^\n')

    # Parse one
    # datadir_name = "../data/pdb_em"
    # dirname = "6NQD_0485"
    # use_template = True
    # pdb_name, mrc_name = dirname.split("_")
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
    # out_name = "output_dock_in_map.pdb" if use_template else "output_dock_in_map_actual.pdb"
    # out_path = os.path.join(datadir_name, dirname, out_name)
    # all_systems = "../data/csvs/filtered.csv"
    # pdb_selections = get_pdb_selection(csv_in=all_systems, columns=['antibody_selection'])
    # selections = pdb_selections[pdb_name.upper()]
    # res = parse_one(out_path, pdb_path, selections, use_template=use_template)
    # print(res)

    # Parse all
    csv_in = '../data/csvs/filtered.csv'
    use_template = False
    pdb_selections = get_pdb_selection(csv_in=csv_in, columns=['antibody_selection'])
    out_dock = f'../data/csvs/benchmark{"_actual" if not use_template else ""}.csv'
    parsed_out = f'../data/csvs/benchmark{"_actual" if not use_template else ""}_parsed.p'
    parse_all_dockinmap(csv_in=out_dock,
                        parsed_out=parsed_out,
                        pdb_selections=pdb_selections,
                        use_template=use_template)
