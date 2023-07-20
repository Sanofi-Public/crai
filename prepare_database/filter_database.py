import os
import sys

import multiprocessing
from collections import defaultdict

import numpy as np
import pandas as pd
import pymol2
import subprocess
import time
from tqdm import tqdm

PHENIX_VALIDATE = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.validation_cryoem"
PHENIX_DOCK_IN_MAP = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.dock_in_map"

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils.mrc_utils import load_mrc
from utils.pymol_utils import list_id_to_pymol_sel
from utils.python_utils import init


def str_resolution_to_float(str_res, default_value=20):
    # Some res are weird
    try:
        fres = float(str_res)
        if fres < 0.1:
            return default_value
        return fres
    except:
        # some are duplicated
        try:
            fres = float(str_res.split(',')[0])
            if fres < 0.1:
                return default_value
            return fres
        except:
            return default_value


def extract_resolution_from_pdb(pdb_path):
    # This successfully extracts the resolution for all systems but 3 :3JCB, 3JCC and 7RAL
    # that don't have a resolution in the mmcif
    word = '_em_3d_reconstruction.resolution '
    with open(pdb_path, 'r') as fp:
        # read all lines in a list
        lines = fp.readlines()
        for line in lines:
            # check if string present on a current line
            if line.find(word) != -1:
                _, resolution = line.split()
                return float(resolution)


def clean_resolution(datadir_name='../data/pdb_em',
                     csv_in="../data/csvs/mapped.csv",
                     csv_out='../data/csvs/resolution.csv'):
    """
    Sabdab fails to parse certain resolutions
    """
    df = pd.read_csv(csv_in, index_col=0, dtype={'mrc': 'str'})
    pruned = df[['resolution', 'pdb', 'mrc']]
    new_resolutions = []
    for i, row in tqdm(pruned.iterrows(), total=len(pruned)):
        resolution, pdb, mrc = row.values

        # Try to read the resolution with a placeholder default value
        # If we hit this, open the mmcif and get a corrected value
        default_value = 25
        resolution = str_resolution_to_float(resolution, default_value=default_value)
        if resolution == default_value:
            pdb_path = os.path.join(datadir_name, f"{pdb.upper()}_{mrc}", f"{pdb.upper()}.cif")
            try:
                resolution = extract_resolution_from_pdb(pdb_path)
            except:
                print('Failed to fix resolution for :', pdb)
                resolution = default_value
        new_resolutions.append(resolution)
    df['resolution'] = new_resolutions
    df.to_csv(csv_out)


def validate_one(mrc, pdb, sel=None, resolution=4.):
    if mrc is None:
        return 2, f"Failed on file loading for {pdb}"
    try:
        if sel is not None:
            with pymol2.PyMOL() as p:
                p.cmd.feedback("disable", "all", "everything")
                p.cmd.load(pdb, 'toto')
                outname = os.path.join(os.path.dirname(pdb), 'temp.pdb')
                p.cmd.save(outname, f'toto and ({sel})')
        pdb_to_score = pdb if sel is None else outname

        cmd = f'{PHENIX_VALIDATE} {pdb_to_score} {mrc} run="*model_vs_data" resolution={resolution}'
        res = subprocess.run(cmd.split(), capture_output=True)

        if sel is not None:
            os.remove(pdb_to_score)

        returncode = res.returncode
        if returncode > 0:
            return returncode, res.stderr.decode()
        stdout = res.stdout.decode()
        stdout_aslist = list(stdout.splitlines())
        relevant_lines_start = stdout_aslist.index('Map-model CC (overall)')
        relevant_lines = stdout_aslist[relevant_lines_start + 2:relevant_lines_start + 6]
        # We now have 4 values : CC_mask, CC_peaks, CC_box. For now let us just us CC_mask
        cc_mask = float(relevant_lines[0].split()[-1])
        return res.returncode, cc_mask
    except Exception as e:
        return 1, e


def add_validation_score(csv_in, csv_out, datadir_name='../data/pdb_em'):
    # Prepare input list
    df = pd.read_csv(csv_in, index_col=0, dtype={'mrc': 'str'})
    df = df[['pdb', 'Hchain', 'Lchain', 'resolution', 'mrc']]
    to_process = []
    for i, row in df.iterrows():
        pdb, heavy_chain, light_chain, resolution, mrc = row.values
        pdb = pdb.upper()
        system_dir = os.path.join(datadir_name, f"{pdb}_{mrc}")
        pdb_path = os.path.join(system_dir, f"{pdb}.cif")
        mrc_path = os.path.join(system_dir, f"emd_{mrc}.map.gz")
        selection = list_id_to_pymol_sel([heavy_chain, light_chain])
        to_process.append((mrc_path, pdb_path, selection, resolution))

    # Parallel computation
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=24, initializer=init, initargs=(l,), )
    results = pool.starmap(validate_one, tqdm(to_process, total=len(to_process)))

    all_results = []
    all_errors = []
    for i, (return_code, result) in enumerate(results):
        if return_code == 0:
            all_results.append(result)
        else:
            all_results.append(-1)
            all_errors.append((return_code, result))
    df['validation_score'] = all_results
    df.to_csv(csv_out)
    for x in all_errors:
        print(x)


def dock_one(mrc, pdb, sel=None, resolution=4.):
    # nproc 1 try 1 : 100.1s
    # nproc 1 : 96.4 s
    # nproc 4 try 1 : 112.16 s
    # nproc 4 : 109.10 s
    if mrc is None:
        return 2, f"Failed on file loading for {pdb}"
    try:
        # with pymol2.PyMOL() as p:
        #     p.cmd.feedback("disable", "all", "everything")
        #     p.cmd.load(pdb, 'toto')
        #     coords = p.cmd.get_coords('toto')
        #     import numpy as np
        #     from scipy.spatial.transform import Rotation as R
        #     rotated = R.random().apply(coords)
        #     translated = rotated + np.array([10, 20, 30])[None, :]
        #     p.cmd.load_coords(translated, "toto", state=1)
        #     outname = os.path.join(os.path.dirname(pdb), 'rotated.pdb')
        #     p.cmd.save(outname, 'toto')
        pdb_out = os.path.join(os.path.dirname(pdb), 'default.pdb')

        if sel is not None:
            with pymol2.PyMOL() as p:
                p.cmd.feedback("disable", "all", "everything")
                p.cmd.load(pdb, 'toto')
                outname = os.path.join(os.path.dirname(pdb), 'temp.pdb')
                p.cmd.save(outname, f'toto and ({sel})')
                hchain, lchain = sel.split()[1], sel.split()[4]
            pdb_out = os.path.join(os.path.dirname(pdb), f'docked_{hchain}_{lchain}.pdb')
        pdb_to_score = pdb if sel is None else outname

        cmd = f'{PHENIX_DOCK_IN_MAP} {pdb_to_score} {mrc} pdb_out={pdb_out} resolution={resolution}'
        res = subprocess.run(cmd.split(), capture_output=True, timeout=5. * 3600)

        if sel is not None:
            os.remove(pdb_to_score)

        returncode = res.returncode
        if returncode > 0:
            return returncode, res.stderr.decode()
        stdout = res.stdout.decode()
        stdout_aslist = list(stdout.splitlines())
        end_of_list = stdout_aslist[-20:]
        translation_line = next(line for line in end_of_list if line.startswith('TRANS'))
        translation_norm = np.linalg.norm(np.array(translation_line.split()[1:]))
        if translation_norm > 8:
            return 3, "too big translation"
        score_line = next(line for line in end_of_list if line.startswith('Wrote placed'))
        score = float(score_line.split()[8])
        return res.returncode, score
    except TimeoutError as e:
        return 1, e
    except Exception as e:
        return 4, e


def add_docking_score(csv_in, csv_out, datadir_name='../data/pdb_em'):
    # Prepare input list
    df = pd.read_csv(csv_in, index_col=0, dtype={'mrc': 'str'})
    df = df[['pdb', 'Hchain', 'Lchain', 'resolution', 'mrc']]
    to_process = []
    for i, row in df.iterrows():
        pdb, heavy_chain, light_chain, resolution, mrc = row.values
        pdb = pdb.upper()
        system_dir = os.path.join(datadir_name, f"{pdb}_{mrc}")
        pdb_path = os.path.join(system_dir, f"{pdb}.cif")
        mrc_path = os.path.join(system_dir, f"emd_{mrc}.map.gz")
        selection = list_id_to_pymol_sel([heavy_chain, light_chain])
        to_process.append((mrc_path, pdb_path, selection, resolution))

    # Parallel computation
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=24, initializer=init, initargs=(l,), )
    results = pool.starmap(dock_one, tqdm(to_process, total=len(to_process)))

    all_results = []
    all_errors = []
    for i, (return_code, result) in enumerate(results):
        if return_code == 0:
            all_results.append(result)
        else:
            all_results.append(-return_code)
            all_errors.append((return_code, result))
    df['docked_validation_score'] = all_results
    df.to_csv(csv_out)
    for x in all_errors:
        print(x)
    # 1 Timeout + error from dock in map, 2 mrc loading buggy, 3 big norm, 4 something else
    # Analysis of errors :
    # 1 (78)  : Collide but a few timeout and a lot just say they fail to find a good solution
    # 2 (3)   : Some don't exist anymore in PDB (obsolete) : 7n01, 6mf7
    # 3 (671) : Very frequent, I count those as a failure for dock in map
    # 4 (42)  : FileNotFoundError or StopIteration.. mysterious


def filter_csv(in_csv="../data/csvs/docked.csv",
               max_resolution=10.,
               out_csv='../data/csvs/filtered.csv',
               nano=False):
    """
    This goes through a csv of systems, filters it :
    - removes systems with empty antigen chain
    - removes systems with no-antibody chain
    - filters on resolution : <10 A

    :param in_csv:
    :return:
    """
    df = pd.read_csv(in_csv, index_col=0, dtype={'mrc': 'str'})
    pruned = df[['Hchain', 'Lchain', 'antigen_chain', 'resolution']]
    ids_to_keeps = []
    ab_sels = []
    ag_sels = []
    for i, row in pruned.iterrows():
        heavy_chain, light_chain, antigen, resolution = row.values

        # Resolution cutoff
        resolution = str_resolution_to_float(resolution)
        if resolution > max_resolution:
            continue

        # Check for nans : if no antigen, just ignore
        if isinstance(antigen, str):
            list_chain_antigen = [chain.strip() for chain in antigen.split('|')]
            antigen_selection = list_id_to_pymol_sel(list_chain_antigen)

            list_chain_antibody = list()
            if isinstance(heavy_chain, str):
                list_chain_antibody.append(heavy_chain)
            if isinstance(light_chain, str):
                list_chain_antibody.append(light_chain)

            # If only one chain, we still accept it (?)
            if (len(list_chain_antibody) == 2 and not nano) or (len(list_chain_antibody) == 1 and nano):
                antibody_selection = list_id_to_pymol_sel(list_chain_antibody)
                ids_to_keeps.append(i)
                ab_sels.append(antibody_selection)
                ag_sels.append(antigen_selection)
    df_new = df.iloc[ids_to_keeps].copy()
    df_new['antibody_selection'] = ab_sels
    df_new['antigen_selection'] = ag_sels
    df_new.to_csv(out_csv)


def sort_date(x):
    """
    MM/DD/YY => YY/MM/DD
    """
    res = x[-2:] + x[:2] + x[-5:-3]
    return res


def sort_by_date(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    sorter_col = df['date'].apply(sort_date)
    df['sorter'] = sorter_col
    df = df.sort_values(by=['sorter'])
    df.drop(columns=['sorter'])
    df.to_csv(csv_out)
    return df


def split_csv(csv_file="../data/csvs/filtered.csv", out_basename='../data/csvs/filtered', other=None):
    """
    :param csv_file:
    :param out_basename:
    :param other: if not None, respect the data split of another split (to be able to cross train)
    :return:
    """
    df = pd.read_csv(csv_file, index_col=0, dtype={'mrc': 'str'})
    unique_pdb = df["pdb"].unique()

    if other is not None:
        # First read the other pdbs
        csv_file = f'{other}_train.csv'
        df_train = pd.read_csv(csv_file, index_col=0, dtype={'mrc': 'str'})
        csv_file = f'{other}_val.csv'
        df_val = pd.read_csv(csv_file, index_col=0, dtype={'mrc': 'str'})
        csv_file = f'{other}_test.csv'
        df_test = pd.read_csv(csv_file, index_col=0, dtype={'mrc': 'str'})

        # Now remove interesting systems and place them in the right split to allow merging the sets
        # We use dummy dicts instead of sets to have consistent order
        unique_pdb = {x: 0 for x in unique_pdb}
        train = {x: 0 for x in df_train["pdb"].unique() if x in unique_pdb}
        val = {x: 0 for x in df_val["pdb"].unique() if x in unique_pdb}
        test = {x: 0 for x in df_test["pdb"].unique() if x in unique_pdb}

        # Finally split the remaining systems to achieve a given size for each set
        size_train = int(0.7 * len(unique_pdb))
        size_val = int(0.85 * len(unique_pdb)) - int(0.7 * len(unique_pdb))
        filtered_unique = {x: 0 for x in unique_pdb if not (x in train or x in val or x in test)}
        for x in filtered_unique:
            if len(train) < size_train:
                train[x] = 0
                continue
            if len(val) < size_val:
                val[x] = 0
                continue
            test[x] = 0
    else:
        train = unique_pdb[:int(0.7 * len(unique_pdb))]
        val = unique_pdb[int(0.7 * len(unique_pdb)):int(0.85 * len(unique_pdb))]
        test = unique_pdb[int(0.85 * len(unique_pdb)):]
    train_df = df[df["pdb"].isin(train)]
    val_df = df[df["pdb"].isin(val)]
    test_df = df[df["pdb"].isin(test)]
    train_df.to_csv(f'{out_basename}_train.csv')
    val_df.to_csv(f'{out_basename}_val.csv')
    test_df.to_csv(f'{out_basename}_test.csv')
    return


if __name__ == '__main__':
    pass

    nanobodies = True
    if not nanobodies:
        mapped = '../data/csvs/mapped.csv'
        resolution = '../data/csvs/resolution.csv'
        validated = '../data/csvs/validated.csv'
        docked = '../data/csvs/docked.csv'
        filtered = '../data/csvs/filtered.csv'
        out_basename = '../data/csvs/filtered'
        sorted_filtered = '../data/csvs/sorted_filtered.csv'
        sorted_out_basename = '../data/csvs/sorted_filtered'
        other = None
        sorted_other = None
    else:
        mapped = '../data/nano_csvs/mapped.csv'
        resolution = '../data/nano_csvs/resolution.csv'
        validated = '../data/nano_csvs/validated.csv'
        docked = '../data/nano_csvs/docked.csv'
        filtered = '../data/nano_csvs/filtered.csv'
        out_basename = '../data/nano_csvs/filtered'
        other = '../data/csvs/filtered'
        sorted_filtered = '../data/nano_csvs/sorted_filtered.csv'
        sorted_out_basename = '../data/nano_csvs/sorted_filtered'
        sorted_other = '../data/csvs/sorted_filtered'

    # ADD RESOLUTION
    # clean_resolution(csv_in=mapped, csv_out=resolution)

    # VALIDATE AND DOCK
    # pdb = "../data/pdb_em/7LO8_23464/7LO8.cif"
    # mrc = "../data/pdb_em/7LO8_23464/emd_23464.map"
    # sel = 'chain H or chain L'
    # validate_one(pdb=pdb, mrc=mrc, sel=sel)
    # dock_one(pdb=pdb, mrc=mrc, sel=sel, resolution=2.8)
    # add_validation_score(csv_in=resolution, csv_out=validated)
    # add_docking_score(csv_in=validated, csv_out=docked)

    # FILTER AND SPLIT : we do two splitting, one being on sorted systems by date
    filter_csv(in_csv=resolution, out_csv=filtered, nano=nanobodies)
    sort_by_date(csv_in=filtered, csv_out=sorted_filtered)
    split_csv(csv_file=filtered, out_basename=out_basename, other=other)
    split_csv(csv_file=sorted_filtered, out_basename=sorted_out_basename, other=sorted_other)
