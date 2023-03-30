"""
Once the raw data is downloaded :
We first 'carve' the mrc to get a box around the pdb to have lighter mrc files.
    During the process, we optionally filter the values far away from the PDB
Then we need to resample the experimental maps to get a fixed voxel_size value of 1.
"""
import os
import sys

import pandas as pd
from collections import defaultdict

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils.mrc_utils import MRCGrid
from utils.pymol_utils import list_id_to_pymol_sel


def process_csv(csv_file="../data/cleaned.csv"):
    """
    This goes through the SabDab reduced output and filters it :
    - removes systems with empty antigen chain
    - removes systems with no-antibody chain
    - filters on resolution : <10 A
    - splits systems that have several chains into different lines

    :param csv_file:
    :return:
    """
    df = pd.read_csv(csv_file)[['pdb', 'Hchain', 'Lchain', 'antigen_chain', 'resolution']]
    # reduced_pdblist = [name[:4].lower() for name in os.listdir("../data/pdb_em")]
    # df_sub = df[df.pdb.isin(reduced_pdblist)]
    # df_sub.to_csv('../data/reduced_clean.csv')
    pdb_selections = defaultdict(list)
    for i, row in df.iterrows():
        pdb, heavy_chain, light_chain, antigen = row.values
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
                pdb_selections[pdb.upper()].append((antibody_selection, antigen_selection))
    return pdb_selections


def process_database(datadir_name="../data/pdb_em", overwrite=False):
    files_list = os.listdir(datadir_name)

    fail_list = []
    for i, dirname in enumerate(files_list):
        if not i % 10:
            print("Done {}/{} files".format(i, len(files_list)))
        try:
            pdb_name, mrc = dirname.split("_")
            pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
            mrcgz_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map.gz")
            carved_name = os.path.join(datadir_name, dirname, "carved.mrc")
            resampled_name = os.path.join(datadir_name, dirname, "resampled_3.mrc")

            mrc = MRCGrid(mrcgz_path)
            mrc.carve(pdb_name=pdb_path, out_name=carved_name, overwrite=overwrite)
            mrc = MRCGrid(carved_name)
            mrc.resample(out_name=resampled_name, new_voxel_size=3, overwrite=overwrite)
        except Exception as e:
            print(e)
            fail_list.append(dirname)
    print(fail_list)


if __name__ == '__main__':
    pass
    process_csv()
    # process_database(overwrite=True)
    # ['5A8H_3096', '7CZW_30519', '7SJO_25163']
