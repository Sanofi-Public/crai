import os
import sys

import pandas as pd
from torch.utils.data.dataset import Dataset

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from load_data.GridComplex import GridComplex
from load_data.CoordComplex import CoordComplex


class ABDataset(Dataset):

    def __init__(self,
                 data_root="../data/pdb_em",
                 csv_to_read="../data/final.csv",
                 return_grid=True,
                 return_sdf=False,
                 rotate=True,
                 crop=0):
        self.data_root = data_root
        self.df = pd.read_csv(csv_to_read)[["pdb_id", "mrc_id", "dirname", "local_ab_id", "antibody_selection"]]
        self.rotate = rotate
        self.return_sdf = return_sdf
        self.return_grid = return_grid
        self.crop = crop

    def __len__(self):
        return len(self.df)

    def unwrapped_get_item(self, item):
        """
        Just useful to desactivate the try/except for debugging
        """
        row = self.df.loc[item].values
        pdb_id, mrc_id, dirname, local_ab_id, antibody_selection = row
        pdb_path = os.path.join(self.data_root, dirname, f'{pdb_id}.cif')
        mrc_path = os.path.join(self.data_root, dirname, f'resampled_{local_ab_id}_2.mrc')
        if self.return_grid:
            comp = GridComplex(mrc_path=mrc_path,
                               pdb_path=pdb_path,
                               antibody_selection=antibody_selection,
                               rotate=self.rotate,
                               return_sdf=self.return_sdf)
        else:
            comp = CoordComplex(mrc_path=mrc_path,
                                pdb_path=pdb_path,
                                antibody_selection=antibody_selection,
                                rotate=self.rotate,
                                crop=self.crop)
        return dirname, comp

    def __getitem__(self, item):
        row = self.df.loc[item].values
        pdb_id, mrc_id, dirname, local_ab_id, antibody_selection = row
        # return self.unwrapped_get_item(item)
        try:
            return self.unwrapped_get_item(item)
        except Exception as e:
            print(f"Buggy data loading for system : {dirname}, local : {local_ab_id},"
                  f" selection :  {antibody_selection}, {e}")
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
