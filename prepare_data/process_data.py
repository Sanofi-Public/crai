"""
Once the raw data is downloaded :
We first 'carve' the mrc to get a box around the pdb to have lighter mrc files.
    During the process, we optionally filter the values far away from the PDB
Then we need to resample the experimental maps to get a fixed voxel_size value of 1.
"""

import os
import sys

from collections import defaultdict
import csv
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils.mrc_utils import MRCGrid, load_mrc
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
    pdb_name, mrc = dirname.split("_")
    pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    mrcgz_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map.gz")
    carved_name = os.path.join(datadir_name, dirname, "carved.mrc")
    resampled_name = os.path.join(datadir_name, dirname, "resampled_3.mrc")
    # mrc = MRCGrid(mrcgz_path)
    try:
        mrc = load_mrc(mrcgz_path)
        boo = mrc.header.mx.item() != mrc.data.shape[0] or \
              mrc.header.my.item() != mrc.data.shape[1] or \
              mrc.header.mz.item() != mrc.data.shape[2]
        if boo:
            print(f'{dirname} : {mrc.header.mx.item()}')
    except Exception as e:
        print(f"Failed for system : {dirname}")


def parallel_do(datadir_name="../data/pdb_em", ):
    """
    Run just do in parallel
    :param datadir_name:
    :return:
    """
    files_list = os.listdir(datadir_name)
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer=init, initargs=(l,))
    njobs = len(files_list)
    pool.starmap(do_one,
                 tqdm(zip(files_list, [datadir_name, ] * njobs), total=njobs))


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


def process_csv(csv_file="../data/cleaned.csv", max_resolution=10.):
    """
    This goes through the SabDab reduced output and filters it :
    - removes systems with empty antigen chain
    - removes systems with no-antibody chain
    - filters on resolution : <10 A
    - group systems that have several chains

    :param csv_file:
    :return:
    """
    df = pd.read_csv(csv_file)[['pdb', 'Hchain', 'Lchain', 'antigen_chain', 'resolution']]

    # # Get subset
    # reduced_pdblist = [name[:4].lower() for name in os.listdir("../data/pdb_em")]
    # df_sub = df[df.pdb.isin(reduced_pdblist)]
    # df_sub.to_csv('../data/reduced_clean.csv')

    # # Get resolution histogram
    # grouped = df.groupby('pdb').nth(0)
    # all_res = grouped[['resolution']].values.squeeze()
    # float_res = [str_resolution_to_float(str_res) for str_res in all_res]
    # plot_res = [res if res < 20 else 20. for res in float_res]
    # plt.hist(plot_res, bins=np.arange(21))
    # plt.show()
    # filtered_res = [res for res in float_res if res < 10]
    # print(f" Retained {len(filtered_res)} / {len(all_res)} systems based on resolution")

    pdb_selections = defaultdict(list)
    for i, row in df.iterrows():
        pdb, heavy_chain, light_chain, antigen, resolution = row.values

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
                pdb_selections[pdb.upper()].append(
                    (antibody_selection, antigen_selection, heavy_chain, light_chain, antigen, resolution))
    return pdb_selections


def filter_pairwise_copies(pdb_path, sel1, sel2):
    """
    The goal is to remove asymetric units by comparing their coordinates
    :param pdb_path:
    :param sel1:
    :param sel2:
    :return: 1 if they are copies 0 otherwise
    """
    from pymol import cmd
    cmd.reinitialize()
    cmd.load(pdb_path, 'toto')
    sel1 = f'toto  and ({sel1})'
    sel2 = f'toto  and ({sel2})'
    cmd.extract("sel1", sel1)
    cmd.extract("sel2", sel2)
    test = cmd.align(mobile="sel2", target="sel1")
    c1 = cmd.get_coords("sel1")
    c2 = cmd.get_coords("sel2")
    if len(c1) == len(c2):
        max_diff = np.max(c1 - c2)
        if max_diff < 1:
            return 1
    return 0


def filter_copies(pdb_path, pdb_selections):
    # At least one should be kept
    list_to_keep = [pdb_selections.pop()]
    # Now let's try to add some more in this list
    for other in pdb_selections:
        found = False
        for kept in list_to_keep:
            if filter_pairwise_copies(pdb_path=pdb_path, sel1=other[0], sel2=kept[0]):
                found = True
                break
        if not found:
            list_to_keep.append(other)
    return list_to_keep


def process_database(datadir_name="../data/pdb_em", overwrite=False):
    files_list = os.listdir(datadir_name)
    pdb_selections = process_csv()

    skip_list, fail_list = []
    columns = "pdb, local_ab_id, heavy_chain, light_chain, antigen, resolution," \
              " antibody_selection, antigen_selection".split(', ')
    df = pd.DataFrame(columns=columns)
    for i, dirname in enumerate(files_list):
        if not i % 10:
            print("Done {}/{} files".format(i, len(files_list)))
        try:
            pdb_name, mrc = dirname.split("_")
            pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
            mrcgz_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map.gz")

            # Don't compute for systems that got discarded (for instance on resolution)
            sels = pdb_selections[pdb_name]
            if len(sels) == 0:
                skip_list.append(dirname)
                continue
            filtered = filter_copies(pdb_path, sels)

            # Now let us compute output files for each unique antibody in the system.
            # We also give it a unique id.
            mrc = MRCGrid(mrcgz_path)
            local_ab_id = 0
            for antibody in filtered:
                antibody_selection, antigen_selection, \
                    heavy_chain, light_chain, antigen, resolution = antibody
                carved_name = os.path.join(datadir_name, dirname, f"carved_{local_ab_id}.mrc")
                resampled_name = os.path.join(datadir_name, dirname, f"resampled_{local_ab_id}_2.mrc")
                mrc.carve(pdb_path=pdb_path, out_name=carved_name, overwrite=overwrite)
                mrc = MRCGrid(carved_name)
                mrc.resample(out_name=resampled_name, new_voxel_size=2, overwrite=overwrite)
                row = [pdb_name, local_ab_id, heavy_chain, light_chain, antigen,
                       resolution, antibody_selection, antigen_selection]
                df.loc[len(df)] = row
                local_ab_id += 1
        except Exception as e:
            print(e)
            fail_list.append(dirname)
    csv_dump = '../data/final.csv'
    df.to_csv(csv_dump)
    print(fail_list)
    print(skip_list)


if __name__ == '__main__':
    pass
    # parallel_do()
    # 3J3O_5291

    # pdb_selections = process_csv()

    # # Get ones from my local database
    # multi_pdbs = [pdb for pdb, sels in pdb_selections.items() if len(sels) > 1]
    # my_pdbs = os.listdir('../data/pdb_em_large')
    # for pdb_em in my_pdbs:
    #     for pdb in multi_pdbs:
    #         if pdb_em.startswith(pdb):
    #             print(pdb_em)
    # =>5A1Z_2996, 6PZY_20540

    # dirname = '5A1Z_2996' # keeps all 3, no symmetry
    # dirname = '6PZY_20540' # goes from 3 to one, there are indeed isomorphic
    # datadir_name = "../data/pdb_em_large"
    # pdb_name, mrc = dirname.split("_")
    # sels = pdb_selections[pdb_name]
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    # filtered = filter_copies(pdb_path, sels)

    process_database(overwrite=True)
    # ['5A8H_3096', '7CZW_30519', '7SJO_25163']
