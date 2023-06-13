"""
Once the raw data is downloaded :
We first 'carve' the mrc to get a box around the pdb to have lighter mrc files.
    During the process, we optionally filter the values far away from the PDB
Then we need to resample the experimental maps to get a fixed voxel_size value of 1.
"""

import os
import sys
import time

from collections import defaultdict
import multiprocessing

import numpy as np
import pandas as pd
import pymol2
from tqdm import tqdm
import subprocess

# phenix = os.environ['PHENIX']
# PHENIX_VALIDATE = os.path.join(phenix, 'build/bin/phenix.validation_cryoem')
PHENIX_VALIDATE = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.validation_cryoem"
PHENIX_DOCK_IN_MAP = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.dock_in_map"

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
    files_list = os.listdir(datadir_name)
    em_mapping = {pdb_em.split('_')[0]: pdb_em.split('_')[1] for pdb_em in files_list}
    to_process = []
    for i, row in df.iterrows():
        pdb, heavy_chain, light_chain, antigen, resolution = row.values
        pdb = pdb.upper()
        if pdb in em_mapping:
            em = em_mapping[pdb]
            system_dir = os.path.join(datadir_name, f"{pdb}_{em}")
            pdb_path = os.path.join(system_dir, f"{pdb}.cif")
            mrc_path = os.path.join(system_dir, f"emd_{em}.map.gz")
            selection = f'chain {heavy_chain} or chain {light_chain}'
            to_process.append((mrc_path, pdb_path, selection, resolution))
        else:
            to_process.append((None, pdb, None, None))

    # # For reduced computation
    # max_systems = 10
    # to_process = to_process[:max_systems]
    # df = df[:max_systems]

    # Parallel computation
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=24, initializer=init, initargs=(l,), )
    results = pool.starmap(validate_one, tqdm(to_process, total=len(to_process)))
    # print(results)

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
            return 3, -2
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
    files_list = os.listdir(datadir_name)
    em_mapping = {pdb_em.split('_')[0]: pdb_em.split('_')[1] for pdb_em in files_list}
    to_process = []
    for i, row in df.iterrows():
        pdb, heavy_chain, light_chain, antigen, resolution, validated = row.values
        pdb = pdb.upper()
        if pdb in em_mapping:
            em = em_mapping[pdb]
            system_dir = os.path.join(datadir_name, f"{pdb}_{em}")
            pdb_path = os.path.join(system_dir, f"{pdb}.cif")
            mrc_path = os.path.join(system_dir, f"emd_{em}.map.gz")
            selection = f'chain {heavy_chain} or chain {light_chain}'
            to_process.append((mrc_path, pdb_path, selection, resolution))
        else:
            to_process.append((None, pdb, None, None))

    # # For reduced computation
    # max_systems = 10
    # to_process = to_process[:max_systems]
    # df = df[:max_systems]

    # Parallel computation
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=24, initializer=init, initargs=(l,), )
    results = pool.starmap(dock_one, tqdm(to_process, total=len(to_process)))
    # print(results)

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


if __name__ == '__main__':
    pass
    # parallel_do()
    # 3J3O_5291

    raw = '../data/cleaned.csv'
    clean_res = '../data/cleaned_res.csv'
    validated = '../data/validated.csv'
    docked = '../data/docked.csv'
    # clean_resolution(csv_in=raw, csv_out=clean_res)

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
    t0 = time.time()
    add_docking_score(csv_in=validated, csv_out=docked)
    print("done in ", time.time() - t0)

    # pdb_selections = process_csv(csv_file=validated)

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
