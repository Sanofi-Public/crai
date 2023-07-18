"""
Starting from the downloaded raw data and a curated csv :
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
import subprocess
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from prepare_database.filter_database import init
from utils.mrc_utils import MRCGrid


def extract_all(in_path='../data/pdb_em/', overwrite=False):
    filelist = os.listdir(in_path)
    for file in tqdm(filelist):
        pdb, em = file.split('_')
        outpath = os.path.join(in_path, file, f"emd_{em}.map")
        if not os.path.exists(outpath) or overwrite:
            cmd1 = f"gunzip -c {outpath}.gz> {outpath}"
            subprocess.call(cmd1, shell=True)


def crop_one_dirname(dirname, datadir_name, overwrite, resample=True):
    try:
        pdb_name, mrc_name = dirname.split("_")
        pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
        mrc_path = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map")
        file_name = "full_crop_resampled_2.mrc" if resample else "full_crop.mrc"
        file_path = os.path.join(datadir_name, dirname, file_name)

        if not os.path.exists(file_path) or overwrite:
            mrc = MRCGrid.from_mrc(mrc_path)
            carved = mrc.carve(pdb_path=pdb_path, margin=25)
            if resample:
                carved.resample(out_name=file_path, new_voxel_size=2, overwrite=overwrite)
            else:
                carved.save(outname=file_path, overwrite=overwrite)
        return 0, None
    except Exception as e:
        print(e)
        return 1, e


def crop_maps(datadir_name="../data/pdb_em",
              parallel=True,
              overwrite=True):
    """
    The first thing we might want to do is to crop and resample our maps, to get lighter files
    :param datadir_name:
    :param parallel:
    :param overwrite:
    :return:
    """
    files_list = os.listdir(datadir_name)
    skip_list, fail_list = [], []
    if not parallel:
        for i, dirname in enumerate(files_list):
            if not i % 10:
                print("Done {}/{} files".format(i, len(files_list)))
            rescode, msg = crop_one_dirname(dirname=dirname,
                                            datadir_name=datadir_name,
                                            overwrite=overwrite)
            if rescode != 0:
                fail_list.append((dirname, msg))
    else:
        files_list = os.listdir(datadir_name)
        l = multiprocessing.Lock()
        nprocs = max(4, os.cpu_count() - 15)
        pool = multiprocessing.Pool(initializer=init, initargs=(l,), processes=nprocs)
        njobs = len(files_list)
        inputs = zip(files_list,
                     [datadir_name, ] * njobs,
                     [overwrite, ] * njobs,
                     )
        results = pool.starmap(crop_one_dirname, tqdm(inputs, total=njobs))
        for dirname, (rescode, msg) in zip(files_list, results):
            if rescode == 1:
                skip_list.append((dirname, msg))
    print("Failed : ", fail_list)


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


def get_pdb_selection(df=None, csv_in=None, columns=None):
    """
    Takes either a df or csv, groups it based on the pdb id and
       returns the queried colums as a list (for each antibody in the PDB) of lists (of each of the column values)
    :param df:
    :param csv_in:
    :param columns:
    :return:
    """
    # In case of a list of dataframes to aggregate, call self a bunch of times and update dict
    if isinstance(csv_in, list):
        all_pdb_selections = defaultdict(list)
        for individual_csv in csv_in:
            individual_dict = get_pdb_selection(csv_in=individual_csv, columns=columns)
            for key, list_sels in individual_dict.items():
                all_pdb_selections[key].extend(list_sels)
        return all_pdb_selections

    if columns is None:
        columns = ['antibody_selection', 'antigen_selection', 'Hchain', 'Lchain', 'antigen_chain', 'resolution']
    if df is None:
        df = pd.read_csv(csv_in, index_col=0, dtype={'mrc': 'str'})
    pdb_selections = defaultdict(list)
    df = df[['pdb'] + columns]
    for i, row in df.iterrows():
        row = row.values
        pdb, content = row[0], row[1:]
        pdb_selections[pdb.upper()].append(content)
    return pdb_selections


def do_one_chunking(dirname, datadir_name, pdb_selections, overwrite):
    pdb_name, mrc_name = dirname.split("_")
    pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    mrcgz_path = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map.gz")

    try:
        sels = pdb_selections[pdb_name]
        # Don't compute for systems that got discarded (for instance on resolution)
        if len(sels) == 0:
            return 1, dirname
        filtered = filter_copies(pdb_path, sels)

        # Now let us compute output files for each unique antibody in the system.
        # We also give it a unique id.
        mrc = MRCGrid.from_mrc(mrcgz_path, normalize=True)
        local_ab_id = 0
        local_rows = []
        for antibody in filtered:
            antibody_selection, antigen_selection, heavy_chain, light_chain, antigen, resolution = antibody
            resampled_name = os.path.join(datadir_name, dirname, f"resampled_{local_ab_id}_2.mrc")
            angstrom_expand = 10
            expanded_selection = f"(({antibody_selection}) expand {angstrom_expand}) or {antigen_selection}"
            carved = mrc.carve(pdb_path=pdb_path, pymol_sel=expanded_selection, margin=8)
            carved.resample(out_name=resampled_name, new_voxel_size=2, overwrite=overwrite)
            row = [pdb_name, mrc_name, dirname, local_ab_id, heavy_chain, light_chain, antigen,
                   resolution, antibody_selection, antigen_selection]
            local_ab_id += 1
            local_rows.append(row)
    except Exception as e:
        print(e)
        return 2, dirname
    return 0, local_rows


def chunk_around(datadir_name="../data/pdb_em",
                 csv_in="../data/csvs/filtered.csv",
                 csv_dump='../data/chunked.csv',
                 parallel=True,
                 overwrite=False):
    """
    This processing goes through a csv and for each system it scans redundant copies of antibodies and keep crops
    around the others
    :param datadir_name:
    :param csv_in:
    :param csv_dump:
    :param parallel:
    :param overwrite:
    :return:
    """
    df_load = pd.read_csv(csv_in, index_col=0, dtype={'mrc': 'str'})
    pdb_selections_mrc = get_pdb_selection(df_load, columns=['mrc'])
    files_list = [f"{pdb.upper()}_{ems[0][0]}" for pdb, ems in pdb_selections_mrc.items()]
    pdb_selections = get_pdb_selection(df_load)
    skip_list, fail_list = [], []
    columns = "pdb, mrc, dirname, local_ab_id, heavy_chain, light_chain, antigen, resolution," \
              " antibody_selection, antigen_selection".split(', ')
    df = pd.DataFrame(columns=columns)
    if not parallel:
        for i, dirname in enumerate(files_list):
            if not i % 30:
                print("Done {}/{} files".format(i, len(files_list)))
            try:
                error, rows = do_one_chunking(dirname=dirname,
                                              datadir_name=datadir_name,
                                              pdb_selections=pdb_selections,
                                              overwrite=overwrite)
                if not error:
                    for row in rows:
                        df.loc[len(df)] = row
                else:
                    skip_list.append(dirname)
            except Exception as e:
                print(e)
                fail_list.append(dirname)
    else:
        l = multiprocessing.Lock()
        nprocs = max(4, os.cpu_count() - 15)
        pool = multiprocessing.Pool(initializer=init, initargs=(l,), processes=nprocs)
        njobs = len(files_list)
        inputs = zip(files_list,
                     [datadir_name, ] * njobs,
                     [pdb_selections, ] * njobs,
                     [overwrite, ] * njobs,
                     )
        results = pool.starmap(do_one_chunking, tqdm(inputs, total=njobs))
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


if __name__ == '__main__':
    pass
    nanobodies = True
    # extract_all()
    # crop_one_dirname(datadir_name="../data/pdb_em/", dirname="6V4N_21042", overwrite=False)
    # crop_maps(overwrite=False)

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

    # filtered = '../data/csvs/filtered.csv'
    # pdb_selections = get_pdb_selection(filtered)
    # dirname = '7KDE_22820'  # Buggy creation
    # dirname = '7P40_13190'
    # dirname = '7LO8_23464'
    # datadir_name = "../data/pdb_em"
    # pdb_name, mrc = dirname.split("_")
    # sels = pdb_selections[pdb_name]
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    # do_one_chunking(dirname=dirname, datadir_name=datadir_name, pdb_selections=pdb_selections, overwrite=False)

    if not nanobodies:
        chunk_around(csv_in='../data/csvs/filtered_train.csv', csv_dump='../data/csvs/chunked_train.csv',
                     overwrite=True)
        chunk_around(csv_in='../data/csvs/filtered_val.csv', csv_dump='../data/csvs/chunked_val.csv', overwrite=True)
        chunk_around(csv_in='../data/csvs/filtered_test.csv', csv_dump='../data/csvs/chunked_test.csv', overwrite=True)
    else:
        chunk_around(csv_in='../data/nano_csvs/filtered_train.csv', csv_dump='../data/nano_csvs/chunked_train.csv',
                     overwrite=True)
        chunk_around(csv_in='../data/nano_csvs/filtered_val.csv', csv_dump='../data/nano_csvs/chunked_val.csv',
                     overwrite=True)
        chunk_around(csv_in='../data/nano_csvs/filtered_test.csv', csv_dump='../data/nano_csvs/chunked_test.csv',
                     overwrite=True)
