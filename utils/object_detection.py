import os
import pickle
import numpy as np
import pymol2
from scipy.spatial.transform import Rotation

from prepare_database.get_templates import REF_PATH_FV, REF_PATH_FAB, REF_PATH_NANO
from utils.rotation import vector_to_rotation
from utils.mrc_utils import MRCGrid

import cripser


# Array to predictions as rotation/translation
def predict_one_ijk(pred_array, margin=2):
    """
    Zero around a point while respecting border effects
    """
    pred_shape = pred_array.shape
    amax = np.argmax(pred_array)
    i, j, k = np.unravel_index(amax, pred_shape)

    # Zero out corresponding region
    imin, imax = max(0, i - margin), min(i + 1 + margin, pred_shape[0])
    jmin, jmax = max(0, j - margin), min(j + 1 + margin, pred_shape[1])
    kmin, kmax = max(0, k - margin), min(k + 1 + margin, pred_shape[2])
    pred_array[imin:imax, jmin:jmax, kmin:kmax] = 0
    return i, j, k


def nms(pred_loc, n_objects=None, thresh=0.2, use_pd=False):
    """
    From a dense array of predictions, return a set of positions based on a number or a threshold
    """
    if use_pd:
        # Topological persistence diagrams : in a watershed, sort by difference between birth and death values.
        pd = cripser.computePH(1 - pred_loc)
        lifetimes = np.clip(pd[:, 2] - pd[:, 1], 0, 1)
        sorter = np.argsort(-lifetimes)
        sorted_pd = pd[sorter]
        if n_objects is None:
            n_objects = np.sum(lifetimes > thresh)
        ijk_s = np.int_(sorted_pd[:n_objects, 3:6])
    else:
        pred_array = pred_loc.copy()
        ijk_s = []
        if n_objects is None:
            while np.max(pred_array) > thresh:
                i, j, k = predict_one_ijk(pred_array)
                ijk_s.append((i, j, k))
                # Avoid fishy situations:
                if len(ijk_s) > 10:
                    break
        else:
            for i in range(n_objects):
                i, j, k = predict_one_ijk(pred_array)
                ijk_s.append((i, j, k))
        ijk_s = np.int_(np.asarray(ijk_s))
    return ijk_s


def output_to_transforms(out_grid, mrc, n_objects=None, thresh=0.5,
                         outmrc=None, classif_nano=False, default_nano=False, use_pd=False):
    """
    First we need to go from grid, complex -> rotation, translation
    Then we call the second one
    """
    pred_loc = out_grid[0]
    pred_shape = pred_loc.shape
    origin = mrc.origin
    top = origin + mrc.voxel_size * mrc.data.shape
    bin_x = np.linspace(origin[0], top[0], num=pred_shape[0] + 1)
    bin_y = np.linspace(origin[1], top[1], num=pred_shape[1] + 1)
    bin_z = np.linspace(origin[2], top[2], num=pred_shape[2] + 1)

    if outmrc is not None:
        voxel_size = (top - origin) / pred_shape
        mrc_pred = MRCGrid(data=pred_loc, voxel_size=voxel_size, origin=origin)
        mrc_pred.save(outname=outmrc, overwrite=True)
        if classif_nano:
            mrc_pred.save(outname=outmrc.replace("pred.mrc", "pred_nano.mrc"), data=out_grid[-1], overwrite=True)

    # First let's find out the position of the antibody in our prediction
    ijk_s = nms(pred_loc, n_objects=n_objects, thresh=thresh, use_pd=use_pd)

    transforms = list()
    for i, j, k in ijk_s:
        predicted_vector = out_grid[1:, i, j, k]
        offset_x, offset_y, offset_z = predicted_vector[:3]
        x = bin_x[i] + offset_x
        y = bin_y[j] + offset_y
        z = bin_z[k] + offset_z
        translation = np.array([x, y, z])

        # Then cast the angles by normalizing them and inverting the angle->R2 transform
        predicted_rz = predicted_vector[3:6] / np.linalg.norm(predicted_vector[3:6])
        cos_t, sin_t = predicted_vector[6:8] / np.linalg.norm(predicted_vector[6:8])
        t = np.arccos(cos_t)
        if np.sin(t) - sin_t > 0.01:
            t = -t

        # Finally build the resulting rotation
        uz_to_p = vector_to_rotation(predicted_rz)
        rotation = uz_to_p * Rotation.from_rotvec([0, 0, t])
        # Assert that the rz with rotation matches predicted_rz
        # rz = rotation.apply([0, 0, 1])
        if classif_nano:
            nano = predicted_vector[8] > 0.5
        else:
            nano = default_nano
        transforms.append((0, translation, rotation, nano))
    return transforms


# rotation/translation to pdbs
def transforms_to_pdb(transforms, out_name=None):
    """
    Take our template and apply the learnt rotation to it.
    """
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        p.cmd.load(REF_PATH_FAB, 'ref_fv')
        p.cmd.load(REF_PATH_NANO, 'ref_nano')
        for i, (rmsd, translation, rotation, nano) in enumerate(transforms):
            hit = f"result_{i}"
            if nano:
                p.cmd.copy(hit, "ref_nano")
            else:
                p.cmd.copy(hit, "ref_fv")
            coords_ref = p.cmd.get_coords(hit)
            p.cmd.alter(hit, f"chain='{i}'")
            rotated = rotation.apply(coords_ref)
            new_coords = rotated + translation[None, :]
            p.cmd.load_coords(new_coords, hit, state=1)
        if out_name is not None:
            to_save = ' or '.join([f"result_{i}" for i in range(len(transforms))])
            p.cmd.save(out_name, to_save)
    return new_coords


# pdb, sel -> rotation/translation
def pdbsel_to_transforms(pdb_path, antibody_selections, cache=True, recompute=False):
    """
    We want to get the transformation to insert the template at the right spot.
    Pymol "align" returns the solid transform as R (x + t), so to avoid getting huge translations (centering first)
       we have to align the template onto the query, so that it outputs the translation to reach the com
       and the right rotation.
    Therefore, the returned matrix has columns that transform the templates and the largest loss weight should be
    on the last row

    
    :param pdb_path:
    :param antibody_selections: list of sels to proceed with
    :param cache: to avoid recomputations
    :return:
    """
    transforms = []
    if isinstance(antibody_selections, str):
        antibody_selections = [antibody_selections]
    for antibody_selection in antibody_selections:
        first_chain = antibody_selection.split()[1]
        dump_align_name = os.path.join(os.path.dirname(pdb_path), f"pymol_chain{first_chain}.p")
        need_recompute = not (cache and os.path.exists(dump_align_name)) or recompute
        if need_recompute:
            # Compute ground truth alignment i.e. the matrix to transform uz in p
            with pymol2.PyMOL() as p:
                p.cmd.feedback("disable", "all", "everything")
                p.cmd.load(pdb_path, 'in_pdb')
                sel = f'in_pdb and ({antibody_selection})'
                p.cmd.extract("to_align", sel)
                residues_to_align = len(p.cmd.get_model("to_align").get_residues())
                nano = False
                if residues_to_align < 165:
                    p.cmd.load(REF_PATH_NANO, 'ref')
                    nano = True
                elif residues_to_align < 300:
                    # len_fv = len(p.cmd.get_model("ref").get_residues())  # len_fv=237, len_fab=446
                    p.cmd.load(REF_PATH_FV, 'ref')
                else:
                    p.cmd.load(REF_PATH_FAB, 'ref')
                coords_ref = p.cmd.get_coords("ref")

                # Now perform the alignment. The com of the aligned template is the object detection objective
                result_align = p.cmd.align(mobile="ref", target="to_align")
                rmsd = result_align[0]

                # We retrieve the parameters of the transformation, notably the rotation
                transformation_matrix = p.cmd.get_object_matrix('ref')
                transformation_matrix = np.asarray(transformation_matrix).reshape((4, 4))
                translation = transformation_matrix[:3, 3]
                rotation = transformation_matrix[:3, :3]
                rotation = Rotation.from_matrix(rotation)
            if cache:
                pickle.dump((rmsd, translation, rotation, nano), open(dump_align_name, 'wb'))
        else:
            rmsd, translation, rotation, nano = pickle.load(open(dump_align_name, 'rb'))
        transforms.append((rmsd, translation, rotation, nano))
    return transforms
