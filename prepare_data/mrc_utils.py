import os
import sys

import pymol.cmd as cmd
import mrcfile
import numpy as np
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


def load_mrc(mrc, mode='r+'):
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


class MRC_grid():
    """
    This class is just used to factor out the interfacing with MRC files easier
    We loose the utilities developped in the mrcfile package and assume a unit voxel_size for simplicity.
    We always have consistent x,y,z and easy to access class member
    The convention for the axis is not the dominant one for MRC, we set it to be (X, Y, Z)
    so that the shape and origin are aligned
    """

    def __init__(self, MRC_file):
        self.mrc_obj = load_mrc(MRC_file)

        self.voxel_size = np.array(self.mrc_obj.voxel_size[..., None].view(dtype=np.float32))
        assert np.allclose(self.voxel_size, np.ones_like(self.voxel_size), atol=0.01)
        self.origin = np.array(self.mrc_obj.header.origin[..., None].view(dtype=np.float32))
        axis_mapping = (int(self.mrc_obj.header.maps) - 1,
                        int(self.mrc_obj.header.mapr) - 1,
                        int(self.mrc_obj.header.mapc) - 1)

        data = self.mrc_obj.data.copy()
        self.data = np.transpose(data, axes=axis_mapping)


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
