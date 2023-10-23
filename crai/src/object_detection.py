import os
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO
import numpy as np
from scipy.spatial.transform import Rotation

script_dir = os.path.dirname(os.path.realpath(__file__))
REF_PATH_FV = os.path.join(script_dir, 'data', f'reference_fv.pdb')
REF_PATH_FAB = os.path.join(script_dir, 'data', f'reference_fab.pdb')
REF_PATH_NANO = os.path.join(script_dir, 'data', f'reference_nano.pdb')

from rotation import vector_to_rotation
from mrc_utils import MRCGrid


# Array to predictions as rotation/translation
def predict_one_ijk(pred_array, margin=3):
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
        import cripser
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
        # rotation = uz_to_p * Rotation.from_rotvec([0, t, 0])
        # Assert that the rz with rotation matches predicted_rz
        # rz = rotation.apply([0, 0, 1])
        if classif_nano:
            nano = predicted_vector[8] > 0.5
        else:
            nano = default_nano
        transforms.append((0, translation, rotation, nano))
    return transforms


# rotation/translation to pdbs
def transforms_to_pdb_biopython(transforms, out_name=None):
    """
    Take our template and apply the learnt rotation to it.
    """
    parser = PDBParser()
    structure_fv = parser.get_structure("fv", REF_PATH_FV)
    structure_nano = parser.get_structure("nano", REF_PATH_NANO)

    coords_fv = np.stack([atom.coord for atom in structure_fv.get_atoms()])
    coords_nano = np.stack([atom.coord for atom in structure_nano.get_atoms()])
    res_structure = Structure('result')
    res_model = Model('result_model')
    res_structure.add(res_model)
    last_chain = 0
    for i, (rmsd, translation, rotation, nano) in enumerate(transforms):
        if nano:
            new_model = structure_nano[0].copy()
            coords_ref = coords_nano
        else:
            new_model = structure_fv[0].copy()
            coords_ref = coords_fv

        for chain in new_model:
            chain.id = str(last_chain)
            last_chain += 1

        rotated = rotation.apply(coords_ref)
        new_coords = rotated + translation[None, :]
        for atom, new_coord in zip(new_model.get_atoms(), new_coords):
            atom.set_coord(new_coord)

        for chain in new_model:
            res_model.add(chain)

    if out_name is not None:
        io = PDBIO()
        io.set_structure(res_structure)
        io.save(out_name)


if __name__ == '__main__':
    transforms_to_pdb_biopython(transforms=[(0, 0, 0, 0), (0, 0, 0, 0)], out_name='test.pdb')
