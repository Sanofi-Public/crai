import os
import sys
import time

import numpy as np
import pickle
import pymol2
from scipy.spatial.transform import Rotation

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from prepare_database.get_templates import REF_PATH_FV, REF_PATH_FAB
from utils import mrc_utils, pymol_utils
from utils.learning_utils import Rotor


def transform_template(translations, rotations, out_name=None):
    """
    Take our template and apply the learnt rotation to it.
    """
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        p.cmd.load(REF_PATH_FV, 'ref')
        coords_ref = p.cmd.get_coords("ref")
        for i, (translation, rotation) in enumerate(zip(translations, rotations)):
            hit = f"result_{i}"
            p.cmd.copy(hit, "ref")
            p.cmd.alter(hit, f"chain='{i}'")
            rotated = rotation.apply(coords_ref)
            new_coords = rotated + translation[None, :]
            p.cmd.load_coords(new_coords, hit, state=1)
        if out_name is not None:
            to_save = ' or '.join([f"result_{i}" for i in range(len(translations))])
            p.cmd.save(out_name, to_save)
    return new_coords


def template_align(pdb_path, sel='polymer.protein'):
    """
    We want to get the transformation to insert the template at the right spot.
    Pymol "align" returns the solid transform as R (x + t), so to avoid getting huge translations (centering first)
       we have to align the template onto the query, so that it outputs the translation to reach the com
       and the right rotation.
    Therefore, the returned matrix has columns that transform the templates and the largest weight should be
    on the last row

    :param pdb_path:
    :param sel:
    :return: 
    """
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        p.cmd.load(pdb_path, 'in_pdb')
        sel = f'in_pdb and ({sel})'
        p.cmd.extract("to_align", sel)
        residues_to_align = len(p.cmd.get_model("to_align").get_residues())
        if residues_to_align < 300:
            # len_fv = len(p.cmd.get_model("ref").get_residues())  # len_fv=237, len_fab=446
            p.cmd.load(REF_PATH_FV, 'ref')
        else:
            p.cmd.load(REF_PATH_FAB, 'ref')
        coords_ref = p.cmd.get_coords("ref")

        # Now perform the alignment. The com of the aligned template is the object detection objective
        test = p.cmd.align(mobile="ref", target="to_align")
        rmsd = test[0]

        # We retrieve the parameters of the transformation, notably the rotation
        transformation_matrix = p.cmd.get_object_matrix('ref')
        transformation_matrix = np.asarray(transformation_matrix).reshape((4, 4))
        rotation = transformation_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation)
        translation = transformation_matrix[:3, 3]
        return rmsd, translation, rotation


class CoordComplex:
    """
    Object containing a protein and a density
    The main difficulty arises from the creation of the grid for the output,
      because we need those to align with the input mrc
    """

    def __init__(self, mrc_path, pdb_path, antibody_selection, rotate=True, crop=0, cache=True):
        # First get the MRC data
        self.mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
        self.initial_mrc_origin = self.mrc.origin

        if cache:
            first_chain = antibody_selection.split()[1]
            dump_align_name = os.path.join(os.path.dirname(pdb_path), f"pymol_chain{first_chain}.p")
            if os.path.exists(dump_align_name):
                rmsd, translation, rotation = pickle.load(open(dump_align_name, 'rb'))
            else:
                # Compute ground truth alignment i.e. the matrix to transform uz in p
                rmsd, translation, rotation = template_align(pdb_path=pdb_path,
                                                             sel=antibody_selection)
                pickle.dump((rmsd, translation, rotation), open(dump_align_name, 'wb'))
        else:
            # Compute ground truth alignment i.e. the matrix to transform uz in p
            rmsd, translation, rotation = template_align(pdb_path=pdb_path,
                                                         sel=antibody_selection)

        if rmsd > 5:
            raise ValueError("The RMSD between template and query is suspiciously high")

        # Add data augmentation : a rotation that rotates the data around the origin of the mrc.
        # The original transform is X' = rotation * X + translation
        # With the extra r_tot (a rotation of pi/4 of the voxels around the mrc origin) it becomes :
        #  r_tot * (X'- origin) + origin =
        #  (r_tot * rotation) * X + (r_tot (com - origin) + origin)
        self.rotor = Rotor() if rotate else Rotor(0, 0)
        self.rotation = self.rotor.r_tot * rotation
        self.translation = self.rotor.r_tot.apply(translation - self.initial_mrc_origin) + self.initial_mrc_origin

        # Careful we need to rotate the mrc after since it changes the origin field
        self.mrc = self.mrc.rotate(rotate_around_z=self.rotor.rotate_around_z,
                                   rotate_in_plane=self.rotor.rotate_in_plane)

        self.mrc = self.mrc.random_crop(crop)
        self.input_tensor = self.mrc.data[None, ...]


if __name__ == '__main__':
    pass

    # # Check that template aligns work on a transformed copy
    # antibody_selection = 'chain H or chain L'
    # pdb_path = '../prepare_database/rotated.pdb'
    # template_align(sel=antibody_selection, pdb_path=pdb_path)

    datadir_name = "../data/pdb_em"
    # dirname = '5H37_9575'
    # antibody_selection = 'chain I or chain M'

    # dirname = "7V3L_31683"
    # antibody_selection = 'chain D or chain E'

    # dirname = '6PZY_20540'
    # antibody_selection = 'chain C or chain D'

    dirname = '6JHS_9829'  # offset between pdb and mrc
    # /home/vmallet/projects/crIA-EM/data/pdb_em/6JHS_9829/emd_9829.map
    antibody_selection = 'chain E or chain D'

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
                               antibody_selection=antibody_selection,
                               rotate=False,
                               crop=3)
    # To plot: get a rotated version of the mrc and compare it to the rotated template
    comp_coords.mrc.save(outname="rotated.mrc", overwrite=True)

    transform_template(translations=[comp_coords.translation],
                       rotations=[comp_coords.rotation],
                       out_name="rotated.pdb")

    # Chimerax command to put colored pseudo atom
    # shape sphere center 81.9, 14.9, 44.9 radius 2 color blue
    # extract toto, 6JHS and (chain E or chain D)
    # extract toto, 7JVC and (chain C or chain D)
    # extract toto, 7XOB and (chain P or chain O)
