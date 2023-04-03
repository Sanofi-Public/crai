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
    rmsd = test[0]
    # We choose a low cutoff for RMSD, for some sytems (5A1Z_2996) the RMSD can be
    # very small (0.95)despite lacking symmetry
    if rmsd < 0.05:
        return 1
    return 0
    # c1 = cmd.get_coords("sel1")
    # c2 = cmd.get_coords("sel2")
    # if len(c1) == len(c2):
    #     max_diff = np.max(c1 - c2)
    #     if max_diff < 1:
    #         return 1
    # return 0


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


def do_one_dirname(dirname, datadir_name, pdb_selections, overwrite):
    pdb_name, mrc_name = dirname.split("_")
    pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    mrcgz_path = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map.gz")

    try:
        # Don't compute for systems that got discarded (for instance on resolution)
        sels = pdb_selections[pdb_name]
        if len(sels) == 0:
            return 1, dirname
        filtered = filter_copies(pdb_path, sels)

        # Now let us compute output files for each unique antibody in the system.
        # We also give it a unique id.
        mrc = MRCGrid(mrcgz_path)
        local_ab_id = 0
        local_rows = []
        for antibody in filtered:
            antibody_selection, antigen_selection, \
                heavy_chain, light_chain, antigen, resolution = antibody
            carved_name = os.path.join(datadir_name, dirname, f"carved_{local_ab_id}.mrc")
            resampled_name = os.path.join(datadir_name, dirname, f"resampled_{local_ab_id}_2.mrc")
            angstrom_expand = 10
            expanded_selection = f"(({antibody_selection}) expand {angstrom_expand}) or {antigen_selection}"
            mrc.carve(pdb_path=pdb_path, out_name=carved_name, overwrite=overwrite,
                      pymol_sel=expanded_selection, margin=6)
            mrc = MRCGrid(carved_name)
            mrc.resample(out_name=resampled_name, new_voxel_size=2, overwrite=overwrite)
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

    # do_one_dirname(dirname='5H37_9575', datadir_name='..', pdb_selections=pdb_selections, overwrite=True)
    # return

    skip_list, fail_list = [], []
    columns = "pdb, mrc, dirname, local_ab_id, heavy_chain, light_chain, antigen, resolution," \
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
        pool = multiprocessing.Pool(initializer=init, initargs=(l,), processes=os.cpu_count() - 15)
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
    new_columns = "pdb, mrc, dirname, local_ab_id, heavy_chain, light_chain, antigen, resolution," \
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

    # dirname = '5A1Z_2996'  # keeps all 3, no symmetry
    # dirname = '6PZY_20540'  # goes from 3 to one, there are indeed isomorphic
    # dirname = '6V4N_21042'  # goes from 4 to one, there are indeed isomorphic
    # datadir_name = ".."
    # datadir_name = "../data/pdb_em_large"
    # pdb_name, mrc = dirname.split("_")
    # sels = pdb_selections[pdb_name]
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    # filtered = filter_copies(pdb_path, sels)

    process_database(overwrite=True)
    # correct_db()
    # Skipped : ['7X1M_32944', '8HC5_34652', '8HCA_34657', '7XXL_33506', '3J42_5674', '7ZLJ_14782', '7U8G_26383',
    #            '5H32_9574', '3J3O_5291', '7YKJ_33892', '8DL7_27498', '7USL_26738', '8DZI_27799', '7X90_33062',
    #            '7T3M_25663', '3J30_5580', '8HIK_34819', '7SK7_25175', '7VGR_31977', '8CW9_27024', '6JFH_9811',
    #            '7T0Z_25585', '8DKE_27488', '7ZYI_15024', '5VJ6_8695', '8DKX_27493', '7URF_26711', '7X8Y_33060',
    #            '8DM4_27526', '7UZ5_26879', '7NRH_12544', '3IY4_5109', '7WTI_32787',
    #            '7RU4_24696', '7T5O_25699', '7DK5_30703', '7XCZ_33130', '7WCD_32421', '8DZH_27798', '8DIU_27443',
    #            '8HS2_34983', '7C2S_30278', '8HSC_34993', '7U9O_26402', '6XJA_22204', '8GSC_34231', '6IDK_9650',
    #            '7T9N_25763', '7WRO_32734', '7WO5_32639', '7UZ7_26881', '8HC8_34655', '7TL0_25982', '7ZBU_14591',
    #            '7WTG_32785', '8HC3_34650', '7XDK_33150', '6JFI_9812', '5A7X_3086', '7SJ0_25149', '7L3N_23156',
    #            '7WJZ_32553', '7UZ9_26883', '7WOC_32648', '8HIJ_34818', '7UZB_26885', '7XDB_33142', '7QTI_14141',
    #            '7T0W_25583', '7UPX_26677', '7UZ4_26878', '7WRZ_32738', '7XCK_33123', '7TJQ_25929', '7WBH_32398',
    #            '7ZLK_14783', '7WTH_32786', '5Y0A_6793', '7URE_26710', '7WO7_32641', '3JCC_6543', '3J7E_5994',
    #            '7SJN_25162', '7WH8_32497', '3IY2_5107', '7SK8_25176', '7T17_25606', '7RU5_24697', '3J8W_6184',
    #            '7ZLI_14781', '8AE0_15377', '7XQ8_33390', '7QTK_14143', '5GZR_9542', '3J70_5020', '3JAB_6258',
    #            '8DIM_27440', '7ZLG_14779', '8DM3_27525', '7DWT_30883', '8HII_34817', '7WRY_32737', '7X7T_33047',
    #            '8HC4_34651', '7SK3_25171', '4CKD_2548', '7SWW_25487', '7WSC_32753', '2R6P_1418', '3IYW_5190',
    #            '7RU3_24695', '6XDG_22137', '8DLR_27512', '8GSD_34232', '6IDI_9649', '6IDL_9651', '4UOM_2645',
    #            '7WWJ_32867', '6AVR_7012', '8EMA_27848', '7TN9_26005', '3J2X_5576', '7WTF_32784', '7WHB_32498',
    #            '7X6A_33019', '8EPA_28523', '3J93_6200', '7U0P_26262', '8DL6_27497', '7UPL_26669', '3J2Y_5578',
    #            '5ANY_3144', '5A8H_3096', '8DW2_27749', '3IY7_5112', '3IXY_5102', '7UVL_26813', '8F6F_28882',
    #            '7YR0_34041', '7WOG_32651', '7SK4_25172', '7XDA_33140', '7V23_31635', '8HC7_34654', '3J8Z_5990',
    #            '8DT3_27690', '7YR1_34042', '7WR9_32719', '7WTK_32789', '7V27_31638', '8F6H_28883', '7UZ6_26880',
    #            '7XW7_33493', '7B09_11964', '7WCU_32429', '7XCP_33125', '8HHY_34807', '8GSF_34234', '7QTJ_14142',
    #            '7V24_31636', '8DLW_27518', '7URC_26708', '8DLS_27514', '8F6E_28881', '8DE6_27385', '7WR8_32718',
    #            '7X8W_33059', '7WHD_32499', '8DL8_27499', '8C7H_16460', '8F6J_28885', '7SIX_25148', '7URD_26709',
    #            '3J2Z_5579', '7RAL_24365', '8ADY_15375', '3IXX_5103', '8GSE_34233', '3JCB_6542', '7WWI_32866',
    #            '7X92_33064', '8HCB_34658', '7WJY_32552', '8HC2_34649', '7L6M_23145', '8AE3_15380', '7Z3A_14474',
    #            '3IY0_5105', '8DW3_27750', '7UZA_26884', '7WOB_32647', '7UMM_26605', '4UOK_2655', '7WO4_32638',
    #            '7URA_26707', '8HHZ_34808', '8HC9_34656', '7SJO_25163', '8AE2_15379', '7WRL_32732', '8DUA_27718',
    #            '7Z12_14438', '7WTJ_32788', '7SK5_25173', '7WRJ_32728', '7U0Q_26263', '8A1E_15073', '7WUR_32839',
    #            '7WWK_32868', '7VGS_31978', '7UOV_26655', '3J8V_6121', '7UZ8_26882', '7SWX_25488', '4UIH_2968',
    #            '7URX_26720', '7U9P_26404', '8HC6_34653', '3IY1_5106', '3JBQ_6258', '6AVQ_7011', '3IY3_5108',
    #            '3JBA_6424', '8HHX_34806', '7YAR_33718', '7WK0_32554', '7X91_33063', '8DAD_27270', '3IY6_5111',
    #            '7X8Z_33061', '7WCK_32423', '7SK9_25177', '8DKW_27492', '8D48_27177', '8F6I_28884', '7U0X_26267',
    #            '7UOT_26653', '7ZLH_14780', '7WCP_32427', '8DVD_27735', '7RU8_24699', '8ADZ_15376', '8E7M_27939',
    #            '6AVU_7013', '3IY5_5110', '3J3Z_5673', '7WOA_32646', '7DWU_30884', '7WP0_32665']
    # Failed :  ['7L06_23095', '6CM3_7516', '7XLT_33286', '7LU9_23518', '6EDU_9038']
