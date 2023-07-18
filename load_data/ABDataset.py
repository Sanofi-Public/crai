import os
import sys

import pandas as pd
from torch.utils.data.dataset import Dataset

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from load_data.GridComplex import GridComplex
from load_data.CoordComplex import CoordComplex
from prepare_database.process_data import get_pdb_selection


class ABDataset(Dataset):

    def __init__(self,
                 data_root="../data/pdb_em",
                 all_systems="../data/csvs/filtered.csv",
                 csv_to_read="../data/csvs/chunked_train.csv",
                 return_grid=False,
                 return_sdf=False,
                 rotate=True,
                 full=False,
                 normalize=False,
                 crop=0):

        self.data_root = data_root
        self.pdb_selections = get_pdb_selection(csv_in=all_systems, columns=['antibody_selection'])
        self.full = full

        # First get_df only to get length, but is not passed to each worker as it can cause memory problems
        self.csv_to_read = csv_to_read
        df = self.get_df()
        self.length = len(df)
        self.df = None

        self.rotate = rotate
        self.crop = crop
        self.normalize = normalize or full

        self.return_sdf = return_sdf
        self.return_grid = return_grid

    def get_df(self):
        columns = ["pdb", "dirname"] if self.full else ["pdb", "dirname", "local_ab_id"]
        if isinstance(self.csv_to_read, list):
            all_dfs = []
            for csv in self.csv_to_read:
                local_df = pd.read_csv(csv)[columns]
                all_dfs.append(local_df)
            df = pd.concat(all_dfs)
        else:
            df = pd.read_csv(self.csv_to_read)[columns]

        if self.full:
            df = df.groupby("pdb", as_index=False).nth(0).reset_index(drop=True)
        return df

    def __len__(self):
        return self.length

    def unwrapped_get_item(self, row):
        """
        Just useful to desactivate the try/except for debugging
        """
        if self.full:
            pdb_id, dirname = row
            mrc_name = f'full_crop_resampled_2.mrc'
        else:
            pdb_id, dirname, local_ab_id = row
            mrc_name = f'resampled_{local_ab_id}_2.mrc'
        pdb_path = os.path.join(self.data_root, dirname, f'{pdb_id}.cif')
        mrc_path = os.path.join(self.data_root, dirname, mrc_name)
        antibody_selections = [res[0] for res in self.pdb_selections[pdb_id]]

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
                                normalize=self.normalize,
                                rotate=self.rotate,
                                crop=self.crop)
        return dirname, comp

    def __getitem__(self, item):
        if self.df is None:
            self.df = self.get_df()
        row = self.df.loc[item].values
        # return self.unwrapped_get_item(row)
        try:
            return self.unwrapped_get_item(row)
        except Exception as e:
            return row[0], None


if __name__ == '__main__':
    pass
    dataset = ABDataset(data_root='../data/pdb_em',
                        csv_to_read='../data/csvs/chunked_train_reduced.csv',
                        return_grid=False,
                        rotate=True,
                        crop=2,
                        full=True
                        )
    res = dataset[3]
    print(res[0])
    a = 1
    # print(res)
