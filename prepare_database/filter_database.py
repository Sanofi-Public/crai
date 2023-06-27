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


def init(l):
    global lock
    lock = l


def do_one(dirname, datadir_name):
    """
    Any jobs that needs to be run on the database, such as statistics, to be done in parallel
    :param dirname:
    :param datadir_name:
    :return:
    """
    try:
        pdb_name, mrc = dirname.split("_")
        # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
        pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
        mrcgz_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map.gz")
        carved_name = os.path.join(datadir_name, dirname, "carved.mrc")
        resampled_name = os.path.join(datadir_name, dirname, "resampled_3.mrc")
        # mrc = MRCGrid.from_mrc(mrcgz_path)

        mrc = load_mrc(mrcgz_path)
        boo = mrc.header.mx.item() != mrc.data.shape[0] or \
              mrc.header.my.item() != mrc.data.shape[1] or \
              mrc.header.mz.item() != mrc.data.shape[2]
        if boo:
            print(f'{dirname} : {mrc.header.mx.item()}')
    except Exception as e:
        # print(f"Failed for system : {dirname}")
        return 1, f"Failed for system : {dirname} with error {e}"


def parallel_do(datadir_name="../data/pdb_em", ):
    """
    Run do one in parallel
    :param datadir_name:
    :return:
    """
    files_list = os.listdir(datadir_name)
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer=init, initargs=(l,))
    njobs = len(files_list)
    results = pool.starmap(do_one,
                           tqdm(zip(files_list, [datadir_name, ] * njobs), total=njobs))
    print(results)


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
    df = pd.read_csv(csv_in, index_col=0)
    new_df = pd.DataFrame(columns=df.columns)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        pdb, heavy_chain, light_chain, antigen, resolution, mrc = row.values

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
        new_df.loc[len(new_df)] = pdb, heavy_chain, light_chain, antigen, resolution, mrc
    new_df.to_csv(csv_out)


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
    df = pd.read_csv(csv_in, index_col=0)
    to_process = []
    for i, row in df.iterrows():
        pdb, heavy_chain, light_chain, antigen, resolution, mrc = row.values
        pdb = pdb.upper()
        system_dir = os.path.join(datadir_name, f"{pdb}_{mrc}")
        pdb_path = os.path.join(system_dir, f"{pdb}.cif")
        mrc_path = os.path.join(system_dir, f"emd_{mrc}.map.gz")
        selection = f'chain {heavy_chain} or chain {light_chain}'
        to_process.append((mrc_path, pdb_path, selection, resolution))

    # # For reduced computation
    # max_systems = 10
    # to_process = to_process[:max_systems]
    # df = df[:max_systems]

    # Parallel computation
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=24, initializer=init, initargs=(l,), )
    results = pool.starmap(validate_one, tqdm(to_process, total=len(to_process)))

    # # For sequential computation
    # results = []
    # for i, x in enumerate(to_process):
    #     print(i)
    #     results.append(validate_one(*x))

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
    df = pd.read_csv(csv_in, index_col=0)
    to_process = []
    for i, row in df.iterrows():
        pdb, heavy_chain, light_chain, antigen, resolution, mrc, validated = row.values
        pdb = pdb.upper()
        system_dir = os.path.join(datadir_name, f"{pdb}_{mrc}")
        pdb_path = os.path.join(system_dir, f"{pdb}.cif")
        mrc_path = os.path.join(system_dir, f"emd_{mrc}.map.gz")
        selection = f'chain {heavy_chain} or chain {light_chain}'
        to_process.append((mrc_path, pdb_path, selection, resolution))

    # # For reduced computation
    # max_systems = 10
    # to_process = to_process[:max_systems]
    # df = df[:max_systems]

    # Parallel computation
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=24, initializer=init, initargs=(l,), )
    results = pool.starmap(dock_one, tqdm(to_process, total=len(to_process)))

    # # For sequential computation
    # results = []
    # for i, x in enumerate(to_process):
    #     print(i)
    #     results.append(validate_one(*x))

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


def process_csv(in_csv="../data/csvs/docked.csv", max_resolution=10., out_csv='../data/csvs/filtered.csv'):
    """
    This goes through a csv of systems, filters it :
    - removes systems with empty antigen chain
    - removes systems with no-antibody chain
    - filters on resolution : <10 A

    :param in_csv:
    :return:
    """
    df = pd.read_csv(in_csv, index_col=0)
    df_new = pd.DataFrame(columns=df.columns.tolist() + ['antibody_selection', 'antigen_selection'])

    # # Get subset
    # reduced_pdblist = [name[:4].lower() for name in os.listdir("../data/pdb_em")]
    # df_sub = df[df.pdb.isin(reduced_pdblist)]

    # df_sub.to_csv('../data/csvs/reduced_clean.csv')

    # # Get resolution histogram
    # import matplotlib.pyplot as plt
    # import numpy as np
    # grouped = df.groupby('pdb').nth(0)
    # all_res = grouped[['resolution']].values.squeeze()
    # float_res = [str_resolution_to_float(str_res) for str_res in all_res]
    # plot_res = [res if res < 20 else 20. for res in float_res]
    # plt.hist(plot_res, bins=np.arange(21))
    # plt.show()
    # filtered_res = [res for res in float_res if res < 10]
    # print(f" Retained {len(filtered_res)} / {len(all_res)} systems based on resolution")
    for i, row in df.iterrows():
        pdb, heavy_chain, light_chain, antigen, resolution, mrc, val, dock = row.values

        # Resolution cutoff
        resolution = str_resolution_to_float(resolution)
        if resolution > max_resolution:
            continue

        # Check for nans : if no antigen, just ignore
        if isinstance(antigen, str):
            list_chain_antigen = antigen.split('|')
            antigen_selection = list_id_to_pymol_sel(list_chain_antigen)

            list_chain_antibody = list()
            if isinstance(heavy_chain, str):
                list_chain_antibody.append(heavy_chain)
            if isinstance(light_chain, str):
                list_chain_antibody.append(light_chain)

            # If only one chain, we still accept it (?)
            if len(list_chain_antibody) > 0:
                antibody_selection = list_id_to_pymol_sel(list_chain_antibody)
                df_new.loc[len(df_new)] = pdb, heavy_chain, light_chain, antigen, resolution, mrc, val, dock, \
                    antibody_selection, antigen_selection
    df_new.to_csv(out_csv)


def split_csv(csv_file="../data/csvs/filtered.csv", out_basename='../data/csvs/filtered'):
    """
    :param csv_file:
    :return:
    """
    df = pd.read_csv(csv_file, index_col=0)
    unique_pdb = df["pdb"].unique()
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
    # parallel_do()
    # 3J3O_5291

    mapped = '../data/csvs/mapped.csv'
    clean_res = '../data/csvs/resolution.csv'
    validated = '../data/csvs/validated.csv'
    docked = '../data/csvs/docked.csv'
    filtered = '../data/csvs/filtered.csv'


    # clean_resolution(csv_in=mapped, csv_out=clean_res)

    # pdb = "../data/pdb_em/7LO8_23464/7LO8.cif"
    # mrc = "../data/pdb_em/7LO8_23464/emd_23464.map"
    # sel = 'chain H or chain L'
    # pdb = '../data/pdb_em/7PC2_13316/7PC2.cif'
    # mrc = '../data/pdb_em/7PC2_13316/emd_13316.map'
    # sel = "chain H or chain G"
    # validate_one(pdb=pdb, mrc=mrc, sel=sel)
    # add_validation_score(csv_in=clean_res, csv_out=validated)

    # res = dock_one(pdb=pdb, mrc=mrc, sel=sel, resolution=2.8)
    # print(res)
    # t0 = time.time()
    # add_docking_score(csv_in=validated, csv_out=docked)
    # print("done in ", time.time() - t0)

    # res = process_csv(docked)
    split_csv()
