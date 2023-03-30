import os
import sys

import pymol.cmd as cmd
import mrcfile
import numpy as np
import scipy
import subprocess


def pymol_parse(pdbname):
    cmd.load(pdbname, "temp_parsing")
    xyz = cmd.get_coords('temp_parsing', 1)
    cmd.delete("temp_parsing")
    return xyz


'''
# Just to benchmark the parsing time against pymol.
# To use coordinates, pymol is 10 times faster.

from Bio.PDB import MMCIFParser
def bio_python(pdbname, parser):
    """
    Just to benchmark against
    """
    structure = parser.get_structure("poulet", pdbname)
    coords = [atom.get_vector() for atom in structure.get_atoms()]
    return coords



# 3.16s vs 0.27 for pymol
a = time.perf_counter()
for i in range(10):
    bio_python(pdbname, parser=parser)
print(f'time1 : {time.perf_counter() - a}')
'''


def load_mrc(mrc, mode='r'):
    """
    returns an mrc from either a mrc or a mrc filename
    :param mrc:
    :param mode:
    :return:
    """
    if isinstance(mrc, str):
        # Buggy support for compressed maps
        if mrc[-3:] == ".gz":
            uncompressed_name = mrc[:-3]
            shell_cmd = f"gunzip -c {mrc} > {uncompressed_name}"
            subprocess.call(shell_cmd, shell=True)
            mrc = mrcfile.open(uncompressed_name, mode=mode)
            os.remove(uncompressed_name)
            return mrc
        mrc = mrcfile.open_async(mrc, mode=mode)
        return mrc.result()
    elif isinstance(mrc, mrcfile.mrcfile.MrcFile):
        return mrc
    else:
        raise ValueError("Wrong input to the MRC loading function")


def save_canonical_mrc(outname, data, voxel_size, origin, overwrite=False):
    """
    Save an mrc with well behaved axis
    Dump an uncompressed file. Chimerax does not handle compressed mrc files
    :param outname: Name of the file
    :param data: data with shape expected to be (x,y,z)
    :param voxel_size: 3 tuple or float
    :param origin: 3 tuple
    :param overwrite: Boolean to overwrite
    :return: None
    """
    with mrcfile.new(outname, overwrite=overwrite) as new_mrc:
        new_mrc.set_data(data)
        new_mrc.header.maps = 1
        new_mrc.header.mapr = 2
        new_mrc.header.mapc = 3
        new_mrc.update_header_from_data()
        new_mrc.update_header_stats()
        voxel_size = ((voxel_size,) * 3) if isinstance(voxel_size, float) else tuple(voxel_size)
        new_mrc.voxel_size = tuple(voxel_size)
        new_mrc.header.origin = tuple(origin)
        # new_mrc.header.mx/nx is done when calling update header from data
        # new_mrc.header.cella is done when updating voxel size


class MRC_grid():
    """
    This class is just used to factor out the interfacing with MRC files
    We loose the utilities developped in the mrcfile package and assume a unit voxel_size for simplicity.
    We always have consistent x,y,z and easy to access class member
    The convention for the axis is not the dominant one for MRC, we set it to be (X, Y, Z)
    so that the shape and origin are aligned
    """

    def __init__(self, MRC_file):
        self.original_mrc = load_mrc(MRC_file)

        # TODO : look into mx,my,mz
        self.voxel_size = np.array(self.original_mrc.voxel_size[..., None].view(dtype=np.float32))

        # The data and the 'x,y,z' annotation might not match.
        # axis_mapping tells us which axis of the data matches which 'xyz' dimension :
        # Convention is : x:0,y:1,z:2   numpy indexing order s:0,r:1,c:2
        # {first_axis (s) : X, Y or Z, second axis (r): x/y/z, third axis (c) : x/y/z}
        # reverse axis mapping tells us which x,y,z is where in the data array
        self.axis_mapping = (int(self.original_mrc.header.maps) - 1,
                             int(self.original_mrc.header.mapr) - 1,
                             int(self.original_mrc.header.mapc) - 1)
        self.reverse_axis_mapping = tuple(self.axis_mapping.index(i) for i in range(3))
        data = self.original_mrc.data.copy()
        # self.data = np.transpose(data, axes=self.axis_mapping)
        self.data = np.transpose(data, axes=self.reverse_axis_mapping)

        # Origin tells you where the lower corner lies in the map.
        # In addition, nxstart gives you an offset for the map : We choose to also normalise that.
        # The shift array convention is as crappy as the s,r,c order:
        original_origin = np.array(self.original_mrc.header.origin[..., None].view(dtype=np.float32))
        shift_array = np.array((self.original_mrc.header.nzstart,  # nzstart always correspond to 's'
                                self.original_mrc.header.nystart,  # nystart always correspond to 'r'
                                self.original_mrc.header.nxstart))  # nxstart always correspond to 'c'
        shift_array_xyz = np.array([shift_array[self.reverse_axis_mapping[i]] for i in range(3)])
        self.origin = original_origin + shift_array_xyz * self.voxel_size

    def carve(self, pdb_name, out_name='carved.mrc', padding=4, filter_cutoff=-1, overwrite=False):
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

        # Using the origin, find the corresponding cells in the mrc
        mins_array = ((xyz_min - self.origin) / self.voxel_size).astype(int, casting='unsafe')
        maxs_array = ((xyz_max - self.origin) / self.voxel_size).astype(int, casting='unsafe')
        mins_array = np.max((mins_array, np.zeros_like(mins_array)), axis=0)
        maxs_array = np.min((maxs_array, np.asarray(self.data.shape) - 1), axis=0)
        grouped_bounds = [(i, j) for i, j in zip(mins_array, maxs_array)]

        # Extract those cells, one must be careful because x is not always the columns index
        data = self.data.copy()
        for i in range(3):
            data = np.take(data, axis=i, indices=range(*grouped_bounds[i]))
        data = np.pad(data, pad_width=padding)
        shifted_origin = self.origin + (mins_array - ((padding,) * 3)) * self.voxel_size

        # Optionnaly select only the cells that are at a certain distance to the pdbs
        if filter_cutoff > 0:
            filter_array = np.zeros_like(data)
            for coord in coords:
                coord_array = ((coord - xyz_min) / self.voxel_size + (padding,) * 3).astype(int, casting='unsafe')
                filter_array[tuple(coord_array)] += 1
            filter_array = np.float_(1. - (filter_array > 0.))
            filter_array = scipy.ndimage.distance_transform_edt(filter_array, sampling=self.voxel_size)
            filter_array = np.array([filter_array < filter_cutoff]).astype(np.float32)
            filter_array = np.reshape(filter_array, data.shape)
            data = filter_array * data

        save_canonical_mrc(outname=out_name,
                           data=data,
                           origin=shifted_origin,
                           voxel_size=self.voxel_size,
                           overwrite=overwrite)

    def resample(self, padding=0, out_name='resample.mrc', overwrite=False, new_voxel_size=1.):
        """
            A script to change the voxel size of a mrc
            The main operation is building a linear interpolation model and doing inference over it.
        :param out_name: Name of the mrc to dump
        :param new_voxel_size: either one or 3 numbers
        :return: None
        """
        if os.path.exists(out_name) and not overwrite:
            return

        # Cast int, float, tuples into (3,) array
        new_voxel_size = new_voxel_size * np.ones(3)

        # Create interpolator in real space (in the cella grid) based on the data
        data = self.data.copy()
        cella = self.voxel_size * np.asarray(self.data.shape)
        data_axes = tuple(np.arange(0, cella[i], step=self.voxel_size[i]) for i in range(3))
        new_axes = tuple(np.arange(0, cella[i], step=new_voxel_size[i]) for i in range(3))
        interpolator = scipy.interpolate.RegularGridInterpolator(data_axes,
                                                                 data,
                                                                 method='linear',
                                                                 bounds_error=False,
                                                                 fill_value=0)

        # Now create the new grid based on the new axis and interpolate over it
        x, y, z = np.meshgrid(*new_axes, indexing='ij')
        flat = x.flatten(), y.flatten(), z.flatten()
        new_grid = np.vstack(flat).T
        new_data_grid = interpolator(new_grid).reshape(x.shape).astype(np.float32)

        # Optionnally add padding and thus shift origin
        new_data_grid = np.pad(new_data_grid, pad_width=padding)
        new_origin = self.origin - padding * new_voxel_size

        save_canonical_mrc(outname=out_name,
                           data=new_data_grid,
                           origin=new_origin,
                           voxel_size=new_voxel_size,
                           overwrite=overwrite)

    def save(self, outname, overwrite=False):
        save_canonical_mrc(outname=outname,
                           data=self.data,
                           origin=self.origin,
                           voxel_size=self.voxel_size,
                           overwrite=overwrite)


def save_coords(coords, topology, outfilename, selection=None):
    """
    Save the coordinates to a pdb file
    • coords: coordinates
    • topology: topology
    • outfilename: name of the oupyt pdb
    • selection: Boolean array to select atoms
    """
    object_name = 'struct_save_coords'
    cmd.delete(object_name)
    if selection is None:
        selection = np.ones(len(topology['resids']), dtype=bool)
    for i, coords_ in enumerate(coords):
        if selection[i]:
            name = topology['names'][i]
            resn = topology['resnames'][i]
            resi = topology['resids'][i]
            chain = topology['chains'][i]
            elem = name[0]
            cmd.pseudoatom(object_name,
                           name=name,
                           resn=resn,
                           resi=resi,
                           chain=chain,
                           elem=elem,
                           hetatm=0,
                           segi=chain,
                           pos=list(coords_))
    cmd.save(outfilename, selection=object_name)
    cmd.delete(object_name)


def save_density(density, outfilename, origin, spacing=1, padding=0):
    """
    Save the density file as mrc for the given atomname
    """
    density = density.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(density.T)
        mrc.voxel_size = spacing
        mrc.header['origin']['x'] = origin[0] - padding + .5 * spacing
        mrc.header['origin']['y'] = origin[1] - padding + .5 * spacing
        mrc.header['origin']['z'] = origin[2] - padding + .5 * spacing
        mrc.update_header_from_data()
        mrc.update_header_stats()


if __name__ == '__main__':
    pass
    # import time

    # List of weird files : different shift_arrays
    # 7WLC_32581
    # 7KR5_23002
    # 7LU9_23518
    # 7MHY_23836
    # 8C7H_16460
    # 7MHZ_23837
    # 7F3Q_31434


    datadir_name = "."
    # datadir_name = "data/pdb_em/"

    # dirname = "3IXX_5103"
    # dirname = "7MHY_23836"
    dirname = "7WLC_32581"  # looks ok
    # dirname = '3IXX_5103'  # looks ok

    pdb_name, mrc = dirname.split("_")
    pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.mmtf.gz")
    map_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map")
    aligned_name = os.path.join(datadir_name, dirname, f"aligned.mrc")
    carved_name = os.path.join(datadir_name, dirname, f"carved.mrc")
    resampled_name = os.path.join(datadir_name, dirname, f"resampled_2.mrc")
    # mrc = MRC_grid(map_path)
    # mrc.save(outname=aligned_name, overwrite=True)
    # mrc.carve(pdb_name=pdb_path, out_name=carved_name, overwrite=True, padding=4, filter_cutoff=2)
    mrc = MRC_grid(carved_name)
    mrc.resample(out_name=resampled_name, new_voxel_size=3, overwrite=True)
