"""
Once the raw data is downloaded :
We first 'carve' the mrc to get a box around the pdb to have lighter mrc files.
    During the process, we optionally filter the values far away from the PDB
Then we need to resample the experimental maps to get a fixed voxel_size value of 1.
"""
import os
import sys

import mrcfile
import numpy as np
import scipy.ndimage
import scipy.interpolate

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils.mrc_utils import load_mrc, pymol_parse


def carve(mrc, pdb_name, out_name='carved.mrc', padding=4, filter_cutoff=-1, overwrite=False):
    """
        This goes from full size to a reduced size, centered around a pdb.
        The main steps are :
            - Getting the coordinates to get a box around the PDB
            - Selecting the right voxels in this box
            - Optionally filter out the values further away from filter_cutoff
            - Creating a new mrc with the origin and the size of the 'cell'
    :param mrc: Either name or mrc file
    :param pdb_name: path to the pdb
    :param out_name: path to the output mrc. If the extension is not .mrc but .map, the origin is not read correctly by
        Chimera
    :param padding: does not need to be an integer
    :param filter_cutoff: negative value will skip the filtering step. Otherwise it's a cutoff in Angstroms
    """
    if os.path.exists(out_name) and not overwrite:
        return

    # Get the bounds from the pdb
    coords = pymol_parse(pdbname=pdb_name)
    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)

    # Load the mrc data
    # the data and the 'x,y,z' annotation might not match.
    mrc = load_mrc(mrc, mode='r+')
    voxel_size = mrc.voxel_size[..., None].view(dtype=np.float32)
    voxel_size = np.array(voxel_size)
    origin = mrc.header.origin[..., None].view(dtype=np.float32)
    origin = np.array(origin)

    # axis_mapping tells us which axis of the data matches which 'xyz' dimension :
    # {first_axis (s) : X, Y or Z, second axis (r): x/y/z, third axis (c) : x/y/z}
    # reverse axis mapping tells us which x,y,z is where in the data array
    axis_mapping = (int(mrc.header.maps) - 1,
                    int(mrc.header.mapr) - 1,
                    int(mrc.header.mapc) - 1)
    reverse_axis_mapping = {value: key for key, value in enumerate(axis_mapping)}
    data_shape_xyz = np.array([mrc.data.shape[reverse_axis_mapping[i]] for i in range(3)])

    # The shift array convention is as crappy as the s,r,c order.
    # I don't get it for now
    shift_array = np.array((mrc.header.nzstart,  # nzstart always correspond to 'c'
                            mrc.header.nystart,  # nystart always correspond to 'r'
                            mrc.header.nxstart))  # nxstart always correspond to 's'
    shift_array_xyz = np.array([shift_array[reverse_axis_mapping[i]] for i in range(3)])
    # shift_array_new = np.array((mrc.header.nxstart,  # nxstart always correspond to 's'
    #                             mrc.header.nystart,  # nystart always correspond to 'r'
    #                             mrc.header.nzstart))  # nzstart always correspond to 'c'
    # shift_array_xyz_new = np.array([shift_array_new[reverse_axis_mapping[i]] for i in range(3)])
    # mins_array_new = ((xyz_min - origin) / voxel_size - shift_array_xyz_new).astype(int, casting='unsafe')
    # maxs_array_new = ((xyz_max - origin) / voxel_size - shift_array_xyz_new).astype(int, casting='unsafe')

    # Using the origin, find the corresponding cells in the mrc
    mins_array = ((xyz_min - origin) / voxel_size - shift_array_xyz).astype(int, casting='unsafe')
    maxs_array = ((xyz_max - origin) / voxel_size - shift_array_xyz).astype(int, casting='unsafe')
    mins_array = np.max((mins_array, np.zeros_like(mins_array)), axis=0)
    maxs_array = np.min((maxs_array, data_shape_xyz - 1), axis=0)
    grouped_bounds = [(i, j) for i, j in zip(mins_array, maxs_array)]
    shifted_origin = xyz_min - ((padding,) * 3) * voxel_size

    # Extract those cells, one must be careful because x is not always the columns index
    data = mrc.data.copy()
    for array_axis, data_axis in enumerate(axis_mapping):
        data = np.take(data, axis=array_axis, indices=range(*grouped_bounds[data_axis]))

    data = np.pad(data, pad_width=padding)

    # Optionnaly select only the cells that are at a certain distance to the pdbs
    if filter_cutoff > 0:
        filter_array = np.zeros_like(data)
        for coord in coords:
            new_coord = ((coord - xyz_min) / voxel_size + (padding,) * 3).astype(int, casting='unsafe')
            new_coord_axis = tuple(new_coord[axis_mapping[i]] for i in range(3))
            filter_array[new_coord_axis] += 1

        filter_array = np.float_(1. - (filter_array > 0.))
        filter_array = scipy.ndimage.distance_transform_edt(filter_array, sampling=voxel_size)
        filter_array = np.array([filter_array < filter_cutoff]).astype(np.float32)
        filter_array = np.reshape(filter_array, data.shape)
        data = filter_array * data

    # Dump an uncompressed file. Chimerax does not handle compressed mrc files
    with mrcfile.new(out_name, overwrite=overwrite) as new_mrc:
        data = np.transpose(data, axes=axis_mapping)
        new_mrc.set_data(data)
        new_mrc.header.maps = 1
        new_mrc.header.mapr = 2
        new_mrc.header.mapc = 3
        new_mrc.update_header_from_data()
        new_mrc.update_header_stats()
        new_mrc.voxel_size = tuple(voxel_size)
        new_mrc.header.origin = tuple(shifted_origin)
        # new_mrc.header.mx is done when calling update header from data
        # new_mrc.header.cella.x is done when updating voxel size


def resample(mrc, padding=0, out_name='resample.mrc', overwrite=False, new_voxel_size=1):
    """
        A script to change the voxel size of a mrc to 1A.
        The main operation is building a linear interpolation model and doing inference over it.
    :param mrc: Either string that will be attempted to be loaded or an MRC object
    :param out_name: Name of the mrc to dump
    :return: None
    """
    if os.path.exists(out_name) and not overwrite:
        return
    mrc = load_mrc(mrc, mode='r')
    voxel_size = mrc.voxel_size[..., None].view(dtype=np.float32)
    voxel_size = np.array(voxel_size)
    axis_mapping = {0: int(mrc.header.maps) - 1,
                    1: int(mrc.header.mapr) - 1,
                    2: int(mrc.header.mapc) - 1}
    data = mrc.data.copy()
    data_axes = tuple(np.arange(0, data.shape[i]) * voxel_size[axis_mapping[i]] for i in range(3))
    interpolator = scipy.interpolate.RegularGridInterpolator(data_axes,
                                                             data,
                                                             method='linear',
                                                             bounds_error=False,
                                                             fill_value=0)
    cella = [mrc.header.cella.x, mrc.header.cella.y, mrc.header.cella.z]
    new_axes = tuple(np.arange(0, cella[axis_mapping[i]], step=new_voxel_size) for i in range(3))
    x, y, z = np.meshgrid(*new_axes, indexing='ij')
    flat = x.flatten(), y.flatten(), z.flatten()
    new_grid = np.vstack(flat).T
    new_data_grid = interpolator(new_grid).reshape(x.shape).astype(np.float32)
    new_data_grid = np.pad(new_data_grid, pad_width=padding)

    with mrcfile.new(out_name, overwrite=overwrite) as mrc_out:
        mrc_out.header.cella.x = (new_data_grid.shape[axis_mapping[0]]) * new_voxel_size
        mrc_out.header.cella.y = (new_data_grid.shape[axis_mapping[1]]) * new_voxel_size
        mrc_out.header.cella.z = (new_data_grid.shape[axis_mapping[2]]) * new_voxel_size
        mrc_out.header.origin.x = mrc.header.origin.x - padding * new_voxel_size
        mrc_out.header.origin.y = mrc.header.origin.y - padding * new_voxel_size
        mrc_out.header.origin.z = mrc.header.origin.z - padding * new_voxel_size
        mrc_out.set_data(new_data_grid)
        mrc_out.update_header_from_data()
        mrc_out.update_header_stats()


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
            carved_name = os.path.join(datadir_name, dirname, f"{mrc}_carved.mrc")
            resampled_name = os.path.join(datadir_name, dirname, f"{mrc}_resampled_3.mrc")

            carve(mrc=mrcgz_path, pdb_name=pdb_path, out_name=carved_name, overwrite=overwrite)
            resample(mrc=carved_name, out_name=resampled_name, padding=0, new_voxel_size=3, overwrite=overwrite)
        except Exception as e:
            print(e)
            fail_list.append(dirname)
    print(fail_list)


if __name__ == '__main__':
    pass
    datadir_name = "../data/pdb_em"
    dirname = '3IXX_5103'  # weird symmetry
    # dirname = '3J70_5020'  # crappy resolution + level goes to 8, what to do with those ?
    # dirname = '3JCX_6629'  # looks ok
    # dirname = '3IY2_5107'  # looks ok
    pdb_name, mrc = dirname.split("_")

    # dir_path = os.path.join(datadir_name, dirname)
    # pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    # mrcgz_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map.gz")
    # mrc_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map")
    # carved_name = os.path.join(datadir_name, dirname, f"{mrc}_carved.mrc")
    # resampled_name = os.path.join(datadir_name, dirname, f"{mrc}_resampled_4.mrc")
    # carve(mrc=mrcgz_path, pdb_name=pdb_path, out_name=carved_name, filter_cutoff=6, overwrite=True)
    # resample(mrc=carved_name, out_name=resampled_name, padding=0, new_voxel_size=4, overwrite=True)

    process_database(overwrite=True)
    # ['5A8H_3096', '7CZW_30519', '7SJO_25163']
