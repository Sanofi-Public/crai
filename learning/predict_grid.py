import os
import sys
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils import mrc_utils
from learning.Unet import UnetModel


def predict_grid(mrc_path, model, process=True, out_name=None, overwrite=True):
    mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    if process:
        mrc = mrc.resample().normalize()
    mrc_grid = torch.from_numpy(mrc.data[None, None, ...])
    with torch.no_grad():
        out = model(mrc_grid)[0].numpy()
    if out_name is not None:
        mrc.save(data=out[0], outname=f"{out_name}_antibody.mrc", overwrite=overwrite)
        mrc.save(data=out[1], outname=f"{out_name}_antigen.mrc", overwrite=overwrite)
        mrc.save(data=out[2], outname=f"{out_name}_void.mrc", overwrite=overwrite)
        mrc.save(data=out[3], outname=f"{out_name}_antibody_dist.mrc", overwrite=overwrite)


if __name__ == '__main__':
    pass
    # dataset = ABDataset()
    # point = dataset[17]
    # print(point)

    # systems = process_csv('../data/reduced_clean.csv')
    # print(systems)

    # comp = Complex(mrc='../data/pdb_em/3IXX_5103/5103_carved.mrc',
    #                pdb_path='../data/pdb_em/3IXX_5103/3IXX.mmtf.gz',
    #                pdb_name='3IXX',
    #                antibody_selection='chain G or chain H or chain I or chain J')

    datadir_name = "../data/pdb_em"
    # datadir_name = ".."
    # dirname = '7V3L_31683' # present in train set
    # dirname = '7LO8_23464'  # this is test set
    dirname = '6NQD_0485'
    pdb_name, mrc_name = dirname.split("_")
    # mrc_path, small = os.path.join(datadir_name, dirname, "resampled_0_2.mrc"), True
    mrc_path, small = os.path.join(datadir_name, dirname, f"emd_{mrc_name}.map.gz"), False

    model_name = 'huge'
    model_path = os.path.join('../saved_models', f"{model_name}.pth")
    model = UnetModel(predict_mse=True,
                      out_channels_decoder=128,
                      num_feature_map=24,
                      )
    model.load_state_dict(torch.load(model_path))
    dump_name = f"{model_name}_{'small' if small else 'large'}"
    predict_grid(mrc_path=mrc_path, model=model, out_name=os.path.join(datadir_name, dirname, dump_name))
