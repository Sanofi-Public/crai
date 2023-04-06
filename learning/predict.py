import os
import sys
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils import mrc_utils
from learning.model import UnetModel


def predict(mrc_path, model, out_name=None, overwrite=True):
    mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    mrc_grid = torch.from_numpy(mrc.data[None, None, ...])
    with torch.no_grad():
        out = model(mrc_grid)[0].numpy()
    if out_name is not None:
        mrc.save(data=out[0], outname=f"{out_name}_antibody.mrc", overwrite=overwrite)
        mrc.save(data=out[1], outname=f"{out_name}_antigen.mrc", overwrite=overwrite)
        mrc.save(data=out[2], outname=f"{out_name}_void.mrc", overwrite=overwrite)


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

    datadir_name = ".."
    dirname = '7V3L_31683'
    # dirname = '7LO8_23464'
    pdb_name, mrc_name = dirname.split("_")
    mrc_path = os.path.join(datadir_name, dirname, "resampled_0_2.mrc")

    model_name = 'first'
    model_path = os.path.join('../saved_models', f"{model_name}.pth")
    model = UnetModel()
    model.load_state_dict(torch.load(model_path))
    predict(mrc_path=mrc_path, model=model, out_name=os.path.join(datadir_name, dirname, model_name))
