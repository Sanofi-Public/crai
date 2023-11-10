from chimerax.core.commands import CmdDesc
from chimerax.core.commands import StringArg, IntArg
from chimerax.core.commands import run
from chimerax.map.volume import Volume

import os
import sys

import pathlib
import time
import numpy as np

import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir))

import utils_mrc
from SimpleUnet import SimpleHalfUnetModel
from utils_object_detection import output_to_transforms, transforms_to_pdb_biopython


def get_mrc(map_path, resample=True, crop=0, normalize='max', session=None):
    t0 = time.time()
    print('Loading data')
    # Get it from chimerax
    if map_path.startswith("#"):
        # #1.1 =>(1,1) which is what's keyed by session._models.
        as_tuple = tuple([int(x) for x in map_path[1:].split('.')])
        if not as_tuple in session.models._models:
            print(f"Could not find queried model {map_path}")
            return None
        queried = session.models._models[as_tuple]
        if not isinstance(queried, Volume):
            print(f"Expected the id to refer to map data, got {queried}")
            return None
        # Chimerax also has this transposition
        queried_data = queried.full_matrix().transpose((2, 1, 0))
        mrc = utils_mrc.MRCGrid(data=queried_data,
                                voxel_size=queried.data.step,
                                origin=queried.data.origin,
                                )
    else:
        mrc = utils_mrc.MRCGrid.from_mrc(map_path)
    if resample:
        mrc = mrc.resample()
    mrc = mrc.normalize(normalize_mode=normalize)
    if crop != 0:
        mrc = mrc.crop(*(crop,) * 6)
    else:
        mrc = mrc.crop_large_mrc()
    print('Data loaded in : ', time.time() - t0)
    return mrc


def get_outname(session, map_path, outname=None):
    """
    If no outname is given, try to get a default
    :param map_path:
    :param outname:
    :return:
    """
    if outname is not None:
        suffix = pathlib.Path(outname).suffix
        if suffix not in {'.pdb', '.cif'}:
            outname += '.pdb'
        return outname

    if map_path.startswith("#"):
        default_outname = "crai_predicted.pdb"
    else:
        default_outname = map_path.replace(".mrc", "_predicted.pdb").replace(".map", "_predicted.pdb")
    if not os.path.exists(default_outname):
        return default_outname
    else:
        print("Default name not available, one could not save the prediction. Please add an outname.")
        return None


def predict_coords(session, mrc, outname=None, outmrc=None, n_objects=None, thresh=0.2, default_nano=False,
                   use_pd=False):
    mrc_grid = torch.from_numpy(mrc.data[None, None, ...])
    print('Loading model...')
    model_path = os.path.join(script_dir, "data/ns_final_last.pth")
    model = SimpleHalfUnetModel(classif_nano=True, num_feature_map=32)
    model.load_state_dict(torch.load(model_path))
    print('Loaded model')

    print('Predicting...')
    t0 = time.time()
    with torch.no_grad():
        out = model(mrc_grid)[0].numpy()
    print(f'Done prediction in : {time.time() - t0:.2f}s', out.shape)
    print('Prediction shape', out.shape)

    print('Post-processing...')
    transforms = output_to_transforms(out, mrc, n_objects=n_objects, thresh=thresh, outmrc=outmrc,
                                      classif_nano=True, default_nano=default_nano, use_pd=use_pd)
    transforms_to_pdb_biopython(transforms=transforms, outname=outname)
    print('Done ! Output saved in ', outname)
    return outname


def crai(session, map_path, outname=None, use_pd=False, n_objects=None):
    """

    :param session:
    :param map_path:
    :param outname:
    :param test_arg:
    :return:
    """
    # run(session, f"open src/data/7LO8_resampled.mrc")
    mrc = get_mrc(map_path=map_path, session=session)
    outname = get_outname(outname=outname, map_path=map_path, session=session)
    if outname is None or mrc is None:
        return None
    predict_coords(mrc=mrc, outname=outname, use_pd=use_pd, n_objects=n_objects, session=session)
    run(session, f"open {outname}")


crai_desc = CmdDesc(required=[("map_path", StringArg)],
                    optional=[("outname", StringArg),
                              ("use_pd", StringArg),
                              ("n_objects", IntArg),
                              ],)
