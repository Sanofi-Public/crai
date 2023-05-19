import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from learning.Unet import HalfUnetModel
from load_data.CoordComplex import transform_template
from utils import mrc_utils
from utils.learning_utils import vector_to_rotation


def align_output(out_grid, mrc):
    """
    Given the output, dump a pdb

    First we need to go from grid, complex -> rotation, translation
    Then we call the second one
    """
    pred_loc = out_grid[0]
    pred_shape = pred_loc.shape
    amax = np.argmax(pred_loc)
    i, j, k = np.unravel_index(amax, pred_shape)
    predicted_vector = out_grid[1:, i, j, k]

    # First let's find out the position of the antibody in our prediction
    offset_x, offset_y, offset_z = predicted_vector[:3]
    origin = mrc.origin
    top = origin + mrc.voxel_size * mrc.data.shape
    bin_x = np.linspace(origin[0], top[0], num=pred_shape[0] + 1)
    bin_y = np.linspace(origin[1], top[1], num=pred_shape[1] + 1)
    bin_z = np.linspace(origin[2], top[2], num=pred_shape[2] + 1)
    x = bin_x[i] + offset_x
    y = bin_y[j] + offset_y
    z = bin_z[k] + offset_z

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
    return np.array([x, y, z]), rotation


def predict_grid(mrc_path, model, process=True, out_name=None, overwrite=True):
    mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    if process:
        mrc = mrc.resample().normalize()
    mrc_grid = torch.from_numpy(mrc.data[None, None, ...])
    with torch.no_grad():
        out = model(mrc_grid)[0].numpy()
    translation, rotation = align_output(out, mrc)
    if out_name is not None:
        transform_template(rotation=rotation, translation=translation, out_name=out_name)
    return translation, rotation


if __name__ == '__main__':
    pass

    datadir_name = "../data/pdb_em"
    # datadir_name = ".."
    # dirname = '7V3L_31683' # present in train set
    # dirname = '7LO8_23464'  # this is test set
    dirname = '6NQD_0485'
    pdb_name, mrc_name = dirname.split("_")
    mrc_path, small = os.path.join(datadir_name, dirname, "resampled_0_2.mrc"), True
    # mrc_path, small = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map.gz"), False

    # mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    # fake_out = torch.randn((1, 9, 23, 28, 19))
    # fake_out[0, 0, ...] = torch.sigmoid(fake_out[0, 0, ...])
    # align_output(fake_out, mrc)

    model_name = 'object_best'
    model_path = os.path.join('../saved_models', f"{model_name}.pth")
    model = HalfUnetModel(out_channels_decoder=128,
                          num_feature_map=24,
                          )
    model.load_state_dict(torch.load(model_path))
    dump_name = f"{model_name}_{'small' if small else 'large'}.pdb"

    predict_grid(mrc_path=mrc_path, model=model, out_name=os.path.join(datadir_name, dirname, dump_name))
