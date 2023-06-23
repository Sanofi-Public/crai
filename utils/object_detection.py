import os
import pickle
import numpy as np
import pymol2
from scipy.spatial.transform import Rotation

from prepare_database.get_templates import REF_PATH_FV, REF_PATH_FAB
from utils.rotation import vector_to_rotation
from utils.mrc_utils import MRCGrid


# Array to predictions as rotation/translation
def predict_one_ijk(pred_array, margin=6):
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


def nms(pred_loc, n_objects=None, thresh=0.5):
    """
    From a dense array of predictions, return a set of positions based on a number or a threshold
    """
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
    return ijk_s


def output_to_transforms(out_grid, mrc, n_objects=None, thresh=0.5, outmrc=None):
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

    # First let's find out the position of the antibody in our prediction
    ijk_s = nms(pred_loc, n_objects=n_objects, thresh=thresh)

    translations, rotations = [], []
    for i, j, k in ijk_s:

        predicted_vector = out_grid[1:, i, j, k]
        offset_x, offset_y, offset_z = predicted_vector[:3]
        x = bin_x[i] + offset_x
        y = bin_y[j] + offset_y
        z = bin_z[k] + offset_z
        translation = np.array([x, y, z])

        # Then cast the angles by normalizing them and inverting the angle->R2 transform
        predicted_rz = predicted_vector[3:6] / np.linalg.norm(predicted_vector[3:6])
        cos_t, sin_t = predicted_vector[6:] / np.linalg.norm(predicted_vector[6:])
        t = np.arccos(cos_t)
        if np.sin(t) - sin_t > 0.01:
            t = -t

        # Finally build the resulting rotation
        uz_to_p = vector_to_rotation(predicted_rz)
        rotation = uz_to_p * Rotation.from_rotvec([0, 0, t])
        # Assert that the rz with rotation matches predicted_rz
        # rz = rotation.apply([0, 0, 1])
        translations.append(translation)
        rotations.append(rotation)

    return translations, rotations


# rotation/translation to pdbs
def transform_to_pdb(translations, rotations, out_name=None):
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


# pdb, sel -> rotation/translation
def pdbsel_to_transform(pdb_path, antibody_selections, cache=True, recompute=False):
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
                translation = transformation_matrix[:3, 3]
                rotation = transformation_matrix[:3, :3]
                rotation = Rotation.from_matrix(rotation)
            if cache:
                pickle.dump((rmsd, translation, rotation), open(dump_align_name, 'wb'))
        else:
            rmsd, translation, rotation = pickle.load(open(dump_align_name, 'rb'))
        transforms.append((rmsd, translation, rotation))
    return transforms
