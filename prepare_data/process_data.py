"""
Once the raw data is downloaded :
We first 'carve' the mrc to get a box around the pdb to have lighter mrc files.
    During the process, we optionally filter the values far away from the PDB
Then we need to resample the experimental maps to get a fixed voxel_size value of 1.
"""
import os
import sys

from utils.mrc_utils import MRC_grid


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

            mrc = MRC_grid(mrcgz_path)
            mrc.carve(pdb_name=pdb_path, out_name=carved_name, overwrite=overwrite)
            mrc = MRC_grid(carved_name)
            mrc.resample(out_name=resampled_name, new_voxel_size=3, overwrite=overwrite)
        except Exception as e:
            print(e)
            fail_list.append(dirname)
    print(fail_list)


if __name__ == '__main__':
    pass

    # process_database(overwrite=True)
    # ['5A8H_3096', '7CZW_30519', '7SJO_25163']
