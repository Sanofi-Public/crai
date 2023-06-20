"""
Starting from the downloaded raw data and a curated csv :
- filter bad data points based on resolution and validation values
- filter symmetric copies of antibodies
- carve the mrc to get a box around the pdb to have lighter mrc files.
- resample the experimental maps to get a fixed voxel_size value of 2.
"""

import os
import sys

from collections import defaultdict
import multiprocessing

import pandas as pd
import pymol2
from tqdm import tqdm

from prepare_database.filter_database import init, str_resolution_to_float

# phenix = os.environ['PHENIX']
# PHENIX_VALIDATE = os.path.join(phenix, 'build/bin/phenix.validation_cryoem')
PHENIX_VALIDATE = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.validation_cryoem"
PHENIX_DOCK_IN_MAP = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.dock_in_map"

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils.mrc_utils import MRCGrid
from utils.pymol_utils import list_id_to_pymol_sel


def process_csv(csv_file="../data/cleaned.csv", max_resolution=10.):
    """
    This goes through a csv of systems, filters it :
    - removes systems with empty antigen chain
    - removes systems with no-antibody chain
    - filters on resolution : <10 A

    Then it groups systems that have several chains

    :param csv_file:
    :return:
    """
    df = pd.read_csv(csv_file, index_col=0)

    # # Get subset
    # reduced_pdblist = [name[:4].lower() for name in os.listdir("../data/pdb_em")]
    # df_sub = df[df.pdb.isin(reduced_pdblist)]

    # df_sub.to_csv('../data/reduced_clean.csv')

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
    df = df[["pdb", "Hchain", "Lchain", "antigen_chain", "resolution"]]
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


def get_rmsd_pairsel(pdb_path, pdb_path2=None, sel1='polymer.protein', sel2='polymer.protein'):
    """
    The goal is to remove asymetric units by comparing their coordinates
    :param pdb_path1:
    :param pdb_path2: if None, just use one PDB
    :param sel1:
    :param sel2:
    :return: 1 if they are copies 0 otherwise
    """
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        p.cmd.load(pdb_path, 'toto')
        sel1 = f'toto  and ({sel1})'

        if pdb_path2 is None:
            sel2 = f'toto  and ({sel2})'
        if pdb_path2 is not None:
            p.cmd.load(pdb_path2, 'titi')
            sel2 = f'titi  and ({sel2})'
        sel2 = f'toto  and ({sel2})'
        p.cmd.extract("sel1", sel1)
        p.cmd.extract("sel2", sel2)
        test = p.cmd.align(mobile="sel2", target="sel1")
        rmsd = test[0]
        return rmsd


def filter_copies(pdb_path, pdb_selections):
    # At least one should be kept
    list_to_keep = [pdb_selections.pop()]

    # Now let's try to add some more in this list
    for other in pdb_selections:
        found = False
        for kept in list_to_keep:
            rmsd = get_rmsd_pairsel(pdb_path=pdb_path,
                                    sel1=other[0],
                                    sel2=kept[0])
            # print(rmsd)
            # We choose a low cutoff for RMSD, for some sytems (5A1Z_2996) the RMSD can be
            # very small (0.95) despite lacking symmetry
            if rmsd < 0.75:
                found = True
                break
        if not found:
            list_to_keep.append(other)

    return list_to_keep


def do_one_dirname(dirname, datadir_name, pdb_selections, overwrite):
    pdb_name, mrc_name = dirname.split("_")
    pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    mrcgz_path = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map.gz")

    try:
        # Don't compute for systems that got discarded (for instance on resolution)
        sels = pdb_selections[pdb_name]
        if len(sels) == 0:
            return 1, dirname
        filtered = filter_copies(pdb_path, sels)

        # Now let us compute output files for each unique antibody in the system.
        # We also give it a unique id.
        mrc = MRCGrid.from_mrc(mrcgz_path, normalize=True)
        local_ab_id = 0
        local_rows = []
        for antibody in filtered:
            antibody_selection, antigen_selection, \
                heavy_chain, light_chain, antigen, resolution = antibody
            resampled_name = os.path.join(datadir_name, dirname, f"resampled_{local_ab_id}_2.mrc")
            angstrom_expand = 10
            expanded_selection = f"(({antibody_selection}) expand {angstrom_expand}) or {antigen_selection}"
            carved = mrc.carve(pdb_path=pdb_path, pymol_sel=expanded_selection, margin=8)
            carved.resample(out_name=resampled_name, new_voxel_size=2, overwrite=overwrite)
            row = [pdb_name, mrc_name, dirname, local_ab_id, heavy_chain, light_chain, antigen,
                   resolution, antibody_selection, antigen_selection]
            local_ab_id += 1
            local_rows.append(row)
    except:
        return 2, dirname
    return 0, local_rows


def process_database(datadir_name="../data/pdb_em",
                     csv_in="../data/cleaned.csv",
                     csv_dump='../data/final.csv',
                     parallel=True,
                     overwrite=False):
    files_list = os.listdir(datadir_name)
    pdb_selections = process_csv(csv_file=csv_in)

    # do_one_dirname(dirname='7V3L_31683', datadir_name='..', pdb_selections=pdb_selections, overwrite=True)
    # do_one_dirname(dirname='5H37_9575', datadir_name='..', pdb_selections=pdb_selections, overwrite=True)
    # return

    skip_list, fail_list = [], []
    columns = "pdb_id, mrc_id, dirname, local_ab_id, heavy_chain, light_chain, antigen, resolution," \
              " antibody_selection, antigen_selection".split(', ')
    df = pd.DataFrame(columns=columns)
    if not parallel:
        for i, dirname in enumerate(files_list):
            if not i % 10:
                print("Done {}/{} files".format(i, len(files_list)))
            try:
                success, rows = do_one_dirname(dirname=dirname,
                                               datadir_name=datadir_name,
                                               pdb_selections=pdb_selections,
                                               overwrite=overwrite)
                if success:
                    for row in rows:
                        df.loc[len(df)] = row
                else:
                    skip_list.append(dirname)
            except Exception as e:
                print(e)
                fail_list.append(dirname)
    else:
        files_list = os.listdir(datadir_name)
        l = multiprocessing.Lock()
        nprocs = max(4, os.cpu_count() - 15)
        pool = multiprocessing.Pool(initializer=init, initargs=(l,), processes=nprocs)
        njobs = len(files_list)
        inputs = zip(files_list,
                     [datadir_name, ] * njobs,
                     [pdb_selections, ] * njobs,
                     [overwrite, ] * njobs,
                     )
        results = pool.starmap(do_one_dirname, tqdm(inputs, total=njobs))
        for fail_code, result in results:
            if fail_code == 0:
                for row in result:
                    df.loc[len(df)] = row
            elif fail_code == 1:
                skip_list.append(result)
            elif fail_code == 2:
                fail_list.append(result)
    df.to_csv(csv_dump)
    print(f"Succeeded on {len(df)} systems, {len(skip_list)} skipped, {len(fail_list)} failed")
    print("Skipped : ", skip_list)
    print("Failed : ", fail_list)


def correct_db(csv='../data/final.csv', new_csv='../data/final_corrected.csv', dirpath="../data/pdb_em"):
    new_columns = "pdb_id, mrc_id, dirname, local_ab_id, heavy_chain, light_chain, antigen, resolution," \
                  " antibody_selection, antigen_selection".split(', ')
    new_df = pd.DataFrame(columns=new_columns)
    old_df = pd.read_csv(csv)
    files_list = os.listdir(dirpath)
    mapping = {}
    # Get pdb : mrc, dirname
    for i, dirname in enumerate(files_list):
        pdb_name, mrc = dirname.split("_")
        mapping[pdb_name] = dirname, mrc
    # Fill the new one with missing values
    for i, row in old_df.iterrows():
        row_values = row.values
        index, pdb_name, local_ab_id, heavy_chain, light_chain, antigen, resolution, \
            antibody_selection, antigen_selection = row_values
        mrc, dirname = mapping[pdb_name]
        new_row = [pdb_name, mrc, dirname, local_ab_id, heavy_chain, light_chain, antigen,
                   resolution, antibody_selection, antigen_selection]
        new_df.loc[len(new_df)] = new_row
    # Dump it
    new_df.to_csv(new_csv)


def crop_one_dirname(dirname, datadir_name, overwrite):
    pdb_name, mrc_name = dirname.split("_")
    pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
    mrc_path = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map")
    resampled_name = os.path.join(datadir_name, dirname, f"full_crop_resampled_2.mrc")
    if not os.path.exists(resampled_name) or overwrite:
        mrc = MRCGrid.from_mrc(mrc_path)
        carved = mrc.carve(pdb_path=pdb_path, margin=8)
        carved.resample(out_name=resampled_name, new_voxel_size=2, overwrite=overwrite)


def crop_maps(datadir_name="../data/pdb_em",
              overwrite=False):
    files_list = os.listdir(datadir_name)
    skip_list, fail_list = [], []
    for i, dirname in enumerate(files_list):
        if not i % 10:
            print("Done {}/{} files".format(i, len(files_list)))
        try:
            crop_one_dirname(dirname=dirname,
                             datadir_name=datadir_name,
                             overwrite=overwrite)
        except Exception as e:
            print(e)
            fail_list.append(dirname)
    print("Failed : ", fail_list)


if __name__ == '__main__':
    pass
    pass
    # parallel_do()
    # 3J3O_5291

    raw = '../data/cleaned.csv'
    clean_res = '../data/cleaned_res.csv'
    validated = '../data/validated.csv'
    docked = '../data/docked.csv'
    # pdb_selections = process_csv(csv_file=validated)

    crop_maps()
    # # Get ones from my local database
    # multi_pdbs = [pdb for pdb, sels in pdb_selections.items() if len(sels) > 1]
    # my_pdbs = os.listdir('../data/pdb_em_large')
    # for pdb_em in my_pdbs:
    #     for pdb in multi_pdbs:
    #         if pdb_em.startswith(pdb):
    #             print(pdb_em)
    # =>5A1Z_2996, 6PZY_20540

    # dirname = '5A1Z_2996'  # keeps all 3, no symmetry, min_diff = 0.88
    # dirname = '6PZY_20540'  # goes from 3 to one, there are indeed isomorphic, max_same = 0
    # dirname = '6V4N_21042'  # goes from 4 to one, there are indeed isomorphic, max_same = 0.004
    # dirname = '7K7I_22700'  # goes from 5 to one, there are indeed isomorphic, max_same = 0.27
    # dirname = '7KDE_22820'  # Goes from 2*3 to 2, with a C3 symmetry and 2 different AB
    #                             # in this example : max_same = 0.58 min_diff=1.28
    # dirname = '6USF_20863'  # keeps 2 with rmsd=3, though written as redundant on the PDB

    # datadir_name = ".."
    # datadir_name = "../data/pdb_em"
    # pdb_name, mrc = dirname.split("_")
    # sels = pdb_selections[pdb_name]
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    # filtered = filter_copies(pdb_path, sels)

    # dirname = '7KDE_22820'  # Buggy creation
    # dirname = '7LO8_23464'
    # datadir_name = "../data/pdb_em"
    # pdb_name, mrc = dirname.split("_")
    # sels = pdb_selections[pdb_name]
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    # do_one_dirname(dirname=dirname, datadir_name=datadir_name, pdb_selections=pdb_selections, overwrite=False)

    # process_database(overwrite=True, csv_in=clean_res, csv_dump='../data/cleaned_final.csv')
    # correct_db()
