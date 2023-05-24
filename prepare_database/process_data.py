"""
Once the raw data is downloaded :
We first 'carve' the mrc to get a box around the pdb to have lighter mrc files.
    During the process, we optionally filter the values far away from the PDB
Then we need to resample the experimental maps to get a fixed voxel_size value of 1.
"""

import os
import sys

from collections import defaultdict
import multiprocessing
import pandas as pd
import pymol2
from tqdm import tqdm

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
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
    mrcgz_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map.gz")
    carved_name = os.path.join(datadir_name, dirname, "carved.mrc")
    resampled_name = os.path.join(datadir_name, dirname, "resampled_3.mrc")
    # mrc = MRCGrid.from_mrc(mrcgz_path)
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
                     csv_in="../data/cleaned.csv",
                     csv_out='../data/cleaned_res.csv'):
    """
    Sabdab fails to parse certain resolutions
    """
    files_list = os.listdir(datadir_name)
    pdb_to_file = {file_name.split('_')[0]: file_name for file_name in files_list}

    df = pd.read_csv(csv_in, index_col=0)
    new_df = pd.DataFrame(columns=df.columns)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        pdb, heavy_chain, light_chain, antigen, resolution = row.values

        # Try to read the resolution with a placeholder default value
        # If we hit this, open the mmcif and get a corrected value
        default_value = 25
        resolution = str_resolution_to_float(resolution, default_value=default_value)
        if resolution == default_value:
            pdb = pdb.upper()
            if not pdb in pdb_to_file:
                continue
            pdb_path = os.path.join(datadir_name, pdb_to_file[pdb], f"{pdb}.cif")
            try:
                resolution = extract_resolution_from_pdb(pdb_path)
            except:
                print('Failed to fix resolution for :', pdb)
                resolution = default_value
        new_df.loc[len(new_df)] = pdb, heavy_chain, light_chain, antigen, resolution
    new_df.to_csv(csv_out)


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
    df = pd.read_csv(csv_file, index_col=0)

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

    print(list_to_keep)
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


if __name__ == '__main__':
    pass
    # parallel_do()
    # 3J3O_5291

    raw = '../data/cleaned.csv'
    clean_res = '../data/cleaned_res.csv'
    clean_resolution(csv_in=raw, csv_out=clean_res)
    pdb_selections = process_csv(csv_file=clean_res)

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

    # process_database(overwrite=True)
    # correct_db()
    # Succeeded on 1113 systems, 249 skipped, 0 failed
    # Skipped = ['3JCC_6543', '7DK5_30703', '7WWJ_32867', '3IY4_5109', '3J8Z_5990', '3IYW_5190', '7Z3A_14474',
    #            '8E7M_27939', '3J2X_5576', '3JAB_6258', '7YR0_34041', '7VGR_31977', '7C2S_30278', '3J7E_5994',
    #            '7U9P_26404', '7T17_25606', '7XDK_33150', '8DL8_27499', '7WCP_32427', '7ZLK_14783', '7UPL_26669',
    #            '6AVR_7012', '8EMA_27848', '7V23_31635', '5Y0A_6793', '8AE2_15379', '7ZLI_14781', '3J8W_6184',
    #            '7WO7_32641', '8DAD_27270', '7UZ7_26881', '7QTJ_14142', '3JCB_6542', '7T0W_25583', '6JFH_9811',
    #            '3IY7_5112', '4CKD_2548', '7WOB_32647', '8HC5_34652', '7WWI_32866', '7WP0_32665', '7UZB_26885',
    #            '8HC9_34656', '8HHX_34806', '7SJN_25162', '7DWT_30883', '3IY6_5111', '3IY2_5107', '3J3O_5291',
    #            '5ANY_3144', '7RU5_24697', '7ZYI_15024', '4UOK_2655', '8DKX_27493', '7XXL_33506', '7U8G_26383',
    #            '7WRO_32734', '7NRH_12544', '7XW7_33493', '8F6H_28883', '8HIK_34819', '8HIJ_34818', '8HC3_34650',
    #            '7U9O_26402', '5A7X_3086', '6AVU_7013', '7URF_26711', '7WRY_32737', '3J30_5580', '7T0Z_25585',
    #            '7SJO_25163', '7RU3_24695', '7SJ0_25149', '7SK7_25175', '7WRZ_32738', '7WR8_32718', '7U0Q_26263',
    #            '7TJQ_25929', '8F6I_28884', '7WO5_32639', '8DW3_27750', '3JBQ_6258', '7SK5_25173', '7WOC_32648',
    #            '7XCZ_33130', '7SK4_25172', '7UZ6_26880', '7UZ8_26882', '7WTG_32785', '7L6M_23145', '7WCD_32421',
    #            '7UZA_26884', '7SWW_25487', '7X92_33064', '7URD_26709', '7X1M_32944', '7T5O_25699', '7XCK_33123',
    #            '7T3M_25663', '3J70_5020', '8DM4_27526', '7WCU_32429', '7X8Z_33061', '8DZH_27798', '8HC4_34651',
    #            '7UMM_26605', '4UOM_2645', '7XDA_33140', '8GSE_34233', '7UPX_26677', '7QTI_14141', '6IDK_9650',
    #            '7X90_33062', '7Z12_14438', '7YAR_33718', '8GSF_34234', '8HC8_34655', '7WH8_32497', '7ZLG_14779',
    #            '8GSC_34231', '8DL7_27498', '3J8V_6121', '7TL0_25982', '8HC6_34653', '6JFI_9812', '7WRJ_32728',
    #            '7DWU_30884', '6IDI_9649', '3J42_5674', '7WTH_32786', '3J2Y_5578', '7WHB_32498', '7WTF_32784',
    #            '8AE0_15377', '5H32_9574', '8DKW_27492', '8HSC_34993', '7URE_26710', '7UOT_26653', '7V27_31638',
    #            '7WK0_32554', '8HS2_34983', '3IY0_5105', '8ADZ_15376', '4UIH_2968', '7WUR_32839', '7SK9_25177',
    #            '8DL6_27497', '5A8H_3096', '3IY5_5110', '7RU4_24696', '7SWX_25488', '2R6P_1418', '8DLS_27514',
    #            '8DLW_27518', '7WCK_32423', '7RAL_24365', '6XDG_22137', '7ZLH_14780', '7X6A_33019', '8HCA_34657',
    #            '7B09_11964', '8HII_34817', '7U0X_26267', '3IXX_5103', '6XJA_22204', '3JBA_6424', '6IDL_9651',
    #            '8GSD_34232', '8DM3_27525', '3IY3_5108', '7SIX_25148', '7WWK_32868', '7TN9_26005', '7X91_33063',
    #            '7URA_26707', '7WTI_32787', '7WR9_32719', '8DIU_27443', '8F6J_28885', '7SK8_25176', '8DZI_27799',
    #            '8HHZ_34808', '7ZBU_14591', '7WOG_32651', '7WRL_32732', '8EPA_28523', '3IXY_5102', '8DT3_27690',
    #            '7YKJ_33892', '8DUA_27718', '8A1E_15073', '7T9N_25763', '7XDB_33142', '3IY1_5106', '7UOV_26655',
    #            '8DVD_27735', '7UZ4_26878', '7URX_26720', '8DIM_27440', '7QTK_14143', '7V24_31636', '6AVQ_7011',
    #            '7UZ9_26883', '7WTJ_32788', '8F6F_28882', '7ZLJ_14782', '8C7H_16460', '7U0P_26262', '8CW9_27024',
    #            '7XCP_33125', '7RU8_24699', '7UVL_26813', '7WTK_32789', '7L3N_23156', '8D48_27177', '7X8W_33059',
    #            '7SK3_25171', '7X8Y_33060', '7WHD_32499', '8DW2_27749', '8HC7_34654', '8DE6_27385', '7URC_26708',
    #            '7YR1_34042', '7USL_26738', '3J3Z_5673', '7WBH_32398', '3J2Z_5579', '7X7T_33047', '8HC2_34649',
    #            '5VJ6_8695', '7VGS_31978', '8HCB_34658', '7WJY_32552', '7WO4_32638', '8DKE_27488', '8ADY_15375',
    #            '7WOA_32646', '7WSC_32753', '7UZ5_26879', '3J93_6200', '8HHY_34807', '7WJZ_32553', '5GZR_9542',
    #            '8F6E_28881', '7XQ8_33390', '8AE3_15380', '8DLR_27512']
