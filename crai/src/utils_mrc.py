import os
import sys

import mrcfile
import numpy as np
import scipy
import subprocess

from utils_rotation import Rotor


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
        mrc = mrcfile.open(mrc, mode=mode)
        return mrc
    elif isinstance(mrc, mrcfile.mrcfile.MrcFile):
        return mrc
    else:
        raise ValueError("Wrong input to the MRC loading function")


def save_canonical_mrc(outname, data, voxel_size, origin, overwrite=False):
    """
    Save a mrc with well-behaved axis
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


class MRCGrid:
    """
    This class is just used to factor out the interfacing with MRC files
    We loose the utilities developped in the mrcfile package and assume a unit voxel_size for simplicity.
    We always have consistent x,y,z and easy to access class member
    The convention for the axis is not the dominant one for MRC, we set it to be (X, Y, Z)
    so that the shape and origin are aligned
    """

    def __init__(self, data, voxel_size, origin, normalize_mode=None):
        self.data = data
        self.voxel_size = voxel_size
        self.origin = origin
        self.normalize(normalize_mode)

    @staticmethod
    def from_mrc(mrc_file, normalize=None):
        with mrcfile.open(mrc_file, mode='r') as original_mrc:
            # The mx,my,mz are almost always equal to the data shape, except for 3J30.
            # This does not make a difference.
            # We ignore the nx fields too and copy voxel size that is already in xyz space.
            voxel_size = np.array(original_mrc.voxel_size[..., None].view(dtype=np.float32))

            # The data and the 'x,y,z' annotation might not match.
            # axis_mapping tells us which axis of the data matches which 'xyz' dimension :
            # Convention is : x:0,y:1,z:2   numpy indexing order s:0,r:1,c:2
            # {first_axis (s) : X, Y or Z, second axis (r): x/y/z, third axis (c) : x/y/z}
            # reverse axis mapping tells us which x,y,z is where in the data array
            axis_mapping = (int(original_mrc.header.maps) - 1,
                            int(original_mrc.header.mapr) - 1,
                            int(original_mrc.header.mapc) - 1)
            reverse_axis_mapping = tuple(axis_mapping.index(i) for i in range(3))
            data = original_mrc.data.copy()
            data = np.transpose(data, axes=reverse_axis_mapping)

            # Origin tells you where the lower corner lies in the map.
            # In addition, nxstart gives you an offset for the map : We choose to also normalise that.
            # The shift array convention is as crappy as the s,r,c order:
            original_origin = np.array(original_mrc.header.origin[..., None].view(dtype=np.float32))
            shift_array = np.array((original_mrc.header.nzstart,  # nzstart always correspond to 's'
                                    original_mrc.header.nystart,  # nystart always correspond to 'r'
                                    original_mrc.header.nxstart))  # nxstart always correspond to 'c'
            shift_array_xyz = np.array([shift_array[reverse_axis_mapping[i]] for i in range(3)])
            origin = original_origin + shift_array_xyz * voxel_size
        return MRCGrid(data=data, voxel_size=voxel_size, origin=origin, normalize_mode=normalize)

    def normalize(self, normalize_mode='max'):
        """
        :return:
        """
        if normalize_mode is None:
            return self

        relued = np.maximum(self.data, np.zeros_like(self.data))
        flat = relued.flatten()
        if normalize_mode == 'max':
            threshold = np.max(flat)
        elif normalize_mode == 'centile':
            above_zero = flat[flat > 0.]
            sorted_above_zero = np.sort(above_zero)
            threshold = sorted_above_zero[int(0.95 * len(sorted_above_zero))]
        else:
            raise ValueError("Unsupported normalization")
        relued = relued / threshold
        new_data = np.minimum(relued, np.ones_like(relued))
        self.data = new_data
        return self

    def resample(self, new_voxel_size=2., out_name=None, overwrite=False, padding=0):
        """
        A script to change the voxel size of a mrc
        The main operation is building a linear interpolation model and doing inference over it.
        :param out_name: Name of the mrc to dump
        :param new_voxel_size: either one or 3 numbers
        :param padding: same
        :param overwrite: boolean
        :return: None
        """
        if out_name is not None and os.path.exists(out_name) and not overwrite:
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

        resampled_mrc = MRCGrid(data=new_data_grid,
                                origin=new_origin,
                                voxel_size=new_voxel_size)
        if out_name is not None:
            resampled_mrc.save(outname=out_name, overwrite=overwrite)
        return resampled_mrc

    def rotate(self, rotate_around_z=None, rotate_in_plane=None):
        """
        Rotate the MRC data around
        :param rotate_around_z:
        :param rotate_in_plane:
        :return:
        """
        rotor = Rotor(rotate_around_z=rotate_around_z, rotate_in_plane=rotate_in_plane)

        new_data, new_origin = rotor.rotate_around_origin(tensor=self.data.copy(),
                                                          origin=self.origin.copy(),
                                                          voxel_size=self.voxel_size.copy())
        rotated_mrc = MRCGrid(data=new_data,
                              origin=new_origin,
                              voxel_size=self.voxel_size.copy())
        return rotated_mrc

    def crop(self, dx1, dx2, dy1, dy2, dz1, dz2):
        new_data = self.data.copy()
        new_data = new_data[dx1:-dx2 - 1, dy1:-dy2 - 1, dz1:-dz2 - 1]
        new_origin = self.origin + self.voxel_size * np.array([dx1, dy1, dz1])
        new_voxel_size = self.voxel_size.copy()
        cropped_mrc = MRCGrid(data=new_data,
                              origin=new_origin,
                              voxel_size=new_voxel_size)
        return cropped_mrc

    def random_crop(self, margin=4):
        """
        Cut the box with random left and right cuts, up to a margin
        :param margin: integer
        :return: a new MRC object
        """
        dx1, dx2, dy1, dy2, dz1, dz2 = np.random.randint(0, margin + 1, size=6)
        return self.crop(dx1, dx2, dy1, dy2, dz1, dz2)

    def save(self, outname, data=None, overwrite=False):
        data = self.data if data is None else data
        save_canonical_mrc(outname=outname,
                           data=data,
                           origin=self.origin,
                           voxel_size=self.voxel_size,
                           overwrite=overwrite)

    def save_npz(self, outname, overwrite=False, compressed=True):
        if os.path.exists(outname) and not overwrite:
            print(".npz file already exists")
        else:
            if compressed:
                np.savez_compressed(file=outname, data=self.data, voxel_size=self.voxel_size, origin=self.origin)
            else:
                np.savez(file=outname, data=self.data, voxel_size=self.voxel_size, origin=self.origin)

    @staticmethod
    def from_npz(npz_file, normalize=None):
        npz_archive = np.load(npz_file)
        return MRCGrid(data=npz_archive["data"],
                       voxel_size=npz_archive["voxel_size"],
                       origin=npz_archive["origin"],
                       normalize=normalize)


if __name__ == '__main__':
    pass
    import time

    # List of weird files : different shift_arrays
    # 7WLC_32581
    # 7KR5_23002
    # 7LU9_23518
    # 7MHY_23836
    # 8C7H_16460
    # 7MHZ_23837
    # 7F3Q_31434

    # datadir_name = ".."
    # datadir_name = "data/pdb_em_large/"
    datadir_name = "../data/pdb_em/"

    # dirname = "7WLC_32581"  # looks ok
    # dirname = '3IXX_5103'  # looks ok
    # dirname = "3J3O_5291"  # large offset between pdb and cryoem
    dirname = "6NQD_0485"
    # dirname = "6V4N_21042"

    pdb_name, mrc = dirname.split("_")
    pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
    map_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map")
    aligned_name = os.path.join(datadir_name, dirname, f"aligned.mrc")
    carved_name = os.path.join(datadir_name, dirname, f"carved.mrc")
    resampled_name = os.path.join(datadir_name, dirname, f"resampled_0_2.mrc")
    rotated_name = os.path.join(datadir_name, dirname, f"rotated.mrc")
    # t0 = time.perf_counter()
    # outname = os.path.join(datadir_name, dirname, f"full_cleaned.mrc")
    # mrc = MRCGrid.from_mrc(mrc_file=map_path)
    # mrc = mrc.resample().normalize(normalize_mode='max')
    # mrc.save(outname, overwrite=True)

    mrc = MRCGrid.from_mrc(mrc_file=map_path)
    # print(f'mrc : {time.perf_counter() - t0}')
    # mrc.save_npz('toto.npz', compressed=False)
    # mrc.save_npz('toto_compressed.npz')
    # t0 = time.perf_counter()
    # mrc = MRCGrid.from_npz(npz_file='toto.npz')
    # print(f'nzp : {time.perf_counter() - t0}')
    # t0 = time.perf_counter()
    # mrc = MRCGrid.from_npz(npz_file='toto_compressed.npz')
    # print(f'compressed : {time.perf_counter() - t0}')
    # CONCLUSION : memory is the same, but loading time for mrc << npz~compressed

    mrc.save(outname=aligned_name, overwrite=True)
    carved_mrc = mrc.carve(pdb_path=pdb_path, out_name=carved_name, pymol_sel='chain C or chain D',
                           overwrite=True, padding=4, filter_cutoff=2)
    carved_mrc.normalize()
    resampled_mrc = carved_mrc.resample(out_name=resampled_name, new_voxel_size=2, overwrite=True)
    rotated_mrc = resampled_mrc.rotate(rotate_around_z=1, rotate_in_plane=1)
    rotated_mrc.save(outname='../data/pdb_em/6PZY_20540/rotated.mrc', overwrite=True)
    mrc.save(outname="rotated.mrc", overwrite=True)
