import os
import sys

import pandas as pd
from torch.utils.data.dataset import Dataset

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from load_data.GridComplex import GridComplex
from load_data.CoordComplex import CoordComplex
from prepare_database.process_data import process_csv


class ABDataset(Dataset):

    def __init__(self,
                 data_root="../data/pdb_em",
                 csv_to_read="../data/final.csv",
                 all_systems="../data/validated.csv",
                 return_grid=True,
                 return_sdf=False,
                 rotate=True,
                 full=False,
                 normalize=False,
                 crop=0):
        self.data_root = data_root
        self.csv_to_read = csv_to_read
        df = pd.read_csv(self.csv_to_read)[
            ["pdb_id", "mrc_id", "dirname", "local_ab_id", "antibody_selection"]]
        self.length = len(df)
        self.df = None

        self.rotate = rotate
        self.crop = crop
        self.normalize = normalize

        self.return_sdf = return_sdf
        self.return_grid = return_grid

        self.pdb_selections = process_csv(csv_file=all_systems)
        self.full = full

    def __len__(self):
        return self.length

    def unwrapped_get_item(self, item):
        """
        Just useful to desactivate the try/except for debugging
        """
        if self.df is None:
            self.df = pd.read_csv(self.csv_to_read)[
                ["pdb_id", "mrc_id", "dirname", "local_ab_id", "antibody_selection"]]
        row = self.df.loc[item].values
        pdb_id, mrc_id, dirname, local_ab_id, antibody_selection = row

        antibody_selections = [res[0] for res in self.pdb_selections[pdb_id]]
        pdb_path = os.path.join(self.data_root, dirname, f'{pdb_id}.cif')
        if self.full:
            mrc_name = f'full_crop_resampled_2.mrc'
        else:
            mrc_name = f'resampled_{local_ab_id}_2.mrc'
        mrc_path = os.path.join(self.data_root, dirname, mrc_name)
        if self.return_grid:
            comp = GridComplex(mrc_path=mrc_path,
                               pdb_path=pdb_path,
                               antibody_selection=antibody_selections,
                               rotate=self.rotate,
                               return_sdf=self.return_sdf)
        else:
            comp = CoordComplex(mrc_path=mrc_path,
                                pdb_path=pdb_path,
                                antibody_selections=antibody_selections,
                                # normalize=self.normalize or self.full,
                                rotate=self.rotate,
                                crop=self.crop)
        return dirname, comp

    def __getitem__(self, item):
        # return self.unwrapped_get_item(item)
        try:
            return self.unwrapped_get_item(item)
        except Exception as e:
            # print(f"Buggy data loading for system : {dirname}, local : {local_ab_id},"
            #      f" selection :  {antibody_selection}, {e}")
            return "failed", None


if __name__ == '__main__':
    pass
    dataset = ABDataset(data_root='../data/pdb_em',
                        csv_to_read='../data/reduced_final.csv',
                        return_grid=False,
                        rotate=True,
                        crop=2
                        )
    res = dataset[3]
    print(res[0])
    print(res[1].shape)
    print(res[2])
    a = 1
    # print(res)
