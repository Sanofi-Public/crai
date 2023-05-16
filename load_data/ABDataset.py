import os
import sys

import pandas as pd
from torch.utils.data.dataset import Dataset

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from load_data.GridComplex import GridComplex


class ABDataset(Dataset):

    def __init__(self,
                 data_root="../data/pdb_em",
                 csv_to_read="../data/final.csv",
                 return_sdf=False,
                 rotate=True):
        self.data_root = data_root
        self.df = pd.read_csv(csv_to_read)[["pdb_id", "mrc_id", "dirname", "local_ab_id", "antibody_selection"]]
        self.rotate = rotate
        self.return_sdf = return_sdf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.loc[item].values
        pdb_id, mrc_id, dirname, local_ab_id, antibody_selection = row
        pdb_path = os.path.join(self.data_root, dirname, f'{pdb_id}.cif')
        # pdb_path = os.path.join(self.data_root, dirname, f'{pdb_id}.mmtf.gz')
        mrc_path = os.path.join(self.data_root, dirname, f'resampled_{local_ab_id}_2.mrc')

        try:

            comp = GridComplex(mrc_path=mrc_path,
                               pdb_path=pdb_path,
                               antibody_selection=antibody_selection,
                               rotate=self.rotate,
                               return_sdf=self.return_sdf)
            input_tensor = comp.input_tensor
            target_tensor = comp.target_tensor
            return dirname, input_tensor, target_tensor


        except:
            print(f"Buggy data loading for system : {dirname}")
            return "failed", [], []


if __name__ == '__main__':
    pass
    # point = dataset[17]
    # print(point)

    # systems = process_csv('../data/reduced_clean.csv')
    # print(systems)

    # from Bio.PDB import *
    #
    # parser = MMCIFParser()
    # structure = parser.get_structure("toto", "3J70.cif")
    # chains = list(structure.get_chains())

    # from pymol import cmd
    # cmd.load("3J70.cif", "toto")
    # chains  = cmd.get_chains('toto')
    # print(chains)
    # pass

    dataset = ABDataset(data_root='../data/pdb_em', csv_to_read='../data/reduced_final.csv')
    res = dataset[3]
    print(res[1].shape)
    print(res[2].shape)
    a = 1
    # print(res)
