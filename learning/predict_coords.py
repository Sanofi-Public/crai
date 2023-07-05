import os
import sys

import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from learning.SimpleUnet import SimpleHalfUnetModel
from utils import mrc_utils
from utils.object_detection import output_to_transforms, transform_to_pdb


def predict_coords(mrc_path, model, process=True, outname=None, outmrc=None, n_objects=None, thresh=0.5,
                   overwrite=True):
    mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    if process:
        mrc = mrc.resample().normalize()
    mrc_grid = torch.from_numpy(mrc.data[None, None, ...])
    with torch.no_grad():
        # out = model(mrc_grid)
        # from train_coords import coords_loss
        # from load_data import CoordComplex
        # pdb_path = '../data/pdb_em/6NQD_0485/6NQD.cif'
        # antibody_selection = 'chain C or chain D'
        # comp_coords = CoordComplex.CoordComplex(mrc_path=mrc_path,
        #                                         pdb_path=pdb_path,
        #                                         antibody_selection=antibody_selection,
        #                                         rotate=False,
        #                                         crop=0)
        # comp_coords.mrc.save('../data/pdb_em/6NQD_0485/test_test.mrc')
        # coords_loss(out, comp_coords)
        out = model(mrc_grid)[0].numpy()
    translations, rotations = output_to_transforms(out, mrc, n_objects=n_objects, thresh=thresh, outmrc=outmrc)
    if outname is not None:
        transform_to_pdb(translations=translations, rotations=rotations, out_name=outname)
    return translations, rotations


if __name__ == '__main__':
    pass

    datadir_name = "../data/pdb_em"
    # datadir_name = ".."
    # dirname = '7V3L_31683' # present in train set
    # dirname = '7LO8_23464'  # this is test set
    # dirname = '6BF9_7093'  # this is test set
    # dirname = '8DG9_27419'  # this is test set
    # dirname = '7DCC_30635'  # this is test set
    # dirname = '6NQD_0485'  # this is test set
    dirname = '6VJA_21212'  # this is close Fvs
    pdb_name, mrc_name = dirname.split("_")
    # mrc_path, small = os.path.join(datadir_name, dirname, "resampled_0_2.mrc"), True
    mrc_path, small = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map.gz"), False
    mrc_path, small = os.path.join(datadir_name, dirname, "full_crop_resampled_2.mrc"), False

    # mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    # fake_out = torch.randn((1, 9, 23, 28, 19))
    # fake_out[0, 0, ...] = torch.sigmoid(fake_out[0, 0, ...])
    # align_output(fake_out, mrc)

    # model_name = 'object_best'
    # model_name = 'object_2_best'
    # model_name = 'object_3_best'
    # model_name = 'crop_95'
    # model_name = 'crop_256'
    # model_name = 'focal_332'
    # model_name = 'less_agg_432'
    # model_name = 'multi_train_339'
    # model_name = 'multi_train_861'
    model_name = 'big_train_gamma_last'
    model_path = os.path.join('../saved_models', f"{model_name}.pth")
    # model = HalfUnetModel(out_channels_decoder=128,
    #                       num_feature_map=24,
    #                       )
    model = SimpleHalfUnetModel(in_channels=1,
                                model_depth=4,
                                num_convs=3,
                                max_decode=2,
                                num_feature_map=32)
    model.load_state_dict(torch.load(model_path))
    dump_name = f"{model_name}_{'small' if small else 'large'}.pdb"
    dump_path = os.path.join(datadir_name, dirname, dump_name)
    out_mrc = dump_path.replace(".pdb", "pred.mrc")
    n_objects = 2
    thresh = 0.5
    predict_coords(mrc_path=mrc_path, model=model, outname=dump_path, outmrc=out_mrc,
                   n_objects=n_objects, thresh=thresh)
