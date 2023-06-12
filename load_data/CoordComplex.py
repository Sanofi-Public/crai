import os
import sys

import pickle

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils import mrc_utils
from utils.rotation import Rotor
from utils.object_detection import transform_to_pdb, pdbsel_to_transform


class CoordComplex:
    """
    Object containing a density and its corresponding antibody
       parametrized as translations and rotations
    """

    def __init__(self, mrc_path, pdb_path, antibody_selections, rotate=True, crop=0, cache=True):
        # First get the MRC data, and store the original origin for rotations around it
        self.mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
        self.initial_mrc_origin = self.mrc.origin

        transforms = pdbsel_to_transform(pdb_path, antibody_selections, cache=cache)
        if any([transform[0] > 5 for transform in transforms]):
            raise ValueError("The RMSD between template and query is suspiciously high")

        # Add data augmentation : a rotation that rotates the data around the origin of the mrc.
        # The original transform is X' = rotation * X + translation
        # With the extra r_tot (a rotation of pi/4 of the voxels around the mrc origin) it becomes :
        #  r_tot * (X'- origin) + origin =
        #  (r_tot * rotation) * X + (r_tot (com - origin) + origin)
        self.rotor = Rotor() if rotate else Rotor(0, 0)
        rotated_transforms = []
        for rmsd, translation, rotation in transforms:
            new_trans, new_rot = self.apply_rotational_augment(translation=translation, rotation=rotation)
            rotated_transforms.append((rmsd, new_trans, new_rot))
        transforms = rotated_transforms
        self.transforms = transforms

        self.mrc = self.mrc.rotate(rotate_around_z=self.rotor.rotate_around_z,
                                   rotate_in_plane=self.rotor.rotate_in_plane)

        self.mrc = self.mrc.random_crop(crop)
        self.input_tensor = self.mrc.data[None, ...]

    def apply_rotational_augment(self, translation, rotation):
        """
        This is an additional rotation around the origin.

        Careful we need to rotate the mrc around the first origin, not mrc.origin as it changes
           upon rotation

        :param translation:
        :param rotation:
        :return:
        """
        rotation = self.rotor.r_tot * rotation
        translation = self.rotor.r_tot.apply(translation - self.initial_mrc_origin) + self.initial_mrc_origin
        return translation, rotation


if __name__ == '__main__':
    pass

    # # Check that template aligns work on a transformed copy
    # antibody_selection = 'chain H or chain L'
    # pdb_path = '../prepare_database/rotated.pdb'
    # pdbsel_to_transform(sel=antibody_selection, pdb_path=pdb_path)

    datadir_name = "../data/pdb_em"
    # dirname = '5H37_9575'
    # antibody_selection = 'chain I or chain M'

    # dirname = "7V3L_31683"
    # antibody_selection = 'chain D or chain E'

    dirname = '6PZY_20540'
    antibody_selection = 'chain C or chain D'

    # /home/vmallet/projects/crIA-EM/data/pdb_em/6JHS_9829/emd_9829.map
    # dirname = '6JHS_9829'  # offset between pdb and mrc
    # antibody_selection = 'chain E or chain D'

    pdb_name, mrc_name = dirname.split("_")
    pdb_path = os.path.join(datadir_name, dirname, f"{pdb_name}.cif")
    resampled_path = os.path.join(datadir_name, dirname, f"resampled_0_2.mrc")

    # # If recomputation of resampled is needed
    # mrc_path = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map")
    # mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    # angstrom_expand = 10
    # expanded_selection = f"(({antibody_selection}) expand {angstrom_expand}) "
    # carved = mrc.carve(pdb_path=pdb_path, pymol_sel=expanded_selection, margin=6)
    # carved.resample(out_name=resampled_path, new_voxel_size=2, overwrite=True)

    comp_coords = CoordComplex(mrc_path=resampled_path,
                               pdb_path=pdb_path,
                               antibody_selections=antibody_selection,
                               rotate=False,
                               crop=3)
    # To plot: get a rotated version of the mrc and compare it to the rotated template
    comp_coords.mrc.save(outname="rotated.mrc", overwrite=True)
    rmsd, translation, rotation = comp_coords.transforms[0]
    transform_to_pdb(translations=[translation],
                     rotations=[rotation],
                     out_name="rotated.pdb")

    # Chimerax command to put colored pseudo atom
    # shape sphere center 81.9, 14.9, 44.9 radius 2 color blue
    # extract toto, 6JHS and (chain E or chain D)
    # extract toto, 7JVC and (chain C or chain D)
    # extract toto, 7XOB and (chain P or chain O)
