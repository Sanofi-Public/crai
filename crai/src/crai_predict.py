from chimerax.core.commands import CmdDesc
from chimerax.core.commands import StringArg, IntArg, BoolArg
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


def get_mrc_from_input(session, map_path):
    # If it's a path, open it in Chimerax and get its id.
    if not map_path.startswith("#"):
        output_run = run(session, f"open {map_path}")
        map_id = output_run[0].id_string
    else:
        map_id = map_path[1:]

    # #1.1 =>(1,1) which is what's keyed by session._models.
    as_tuple = tuple([int(x) for x in map_id.split('.')])
    if not as_tuple in session.models._models:
        raise ValueError(f"Could not find queried model {map_path}")
    queried = session.models._models[as_tuple]
    if not isinstance(queried, Volume):
        raise ValueError(f"Expected the id to refer to map data, got {queried}")
    # Chimerax also has this transposition
    queried_data = queried.full_matrix().transpose((2, 1, 0))
    mrc = utils_mrc.MRCGrid(data=queried_data,
                            voxel_size=queried.data.step,
                            origin=queried.data.origin,
                            )
    return map_id, mrc


def clean_mrc(mrc, resample=True, crop=0, normalize='max'):
    if resample:
        mrc = mrc.resample()
    mrc = mrc.normalize(normalize_mode=normalize)
    if crop != 0:
        mrc = mrc.crop(*(crop,) * 6)
    else:
        mrc = mrc.crop_large_mrc()
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
        default_outname = "crai_prediction.pdb"
    else:
        default_outname = map_path.replace(".mrc", "_predicted.pdb").replace(".map", "_predicted.pdb")
    if not os.path.exists(default_outname):
        return default_outname
    else:
        print("Default name not available, one could not save the prediction. Please add an outname.")
        return None


def predict_coords(session, mrc, outname=None, outmrc=None, n_objects=None, thresh=0.2, default_nano=False,
                   use_pd=True, split_pred=True):
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
    outnames = transforms_to_pdb_biopython(transforms=transforms, outname=outname, split_pred=split_pred)
    print('Done ! Output saved in ', outnames[0])
    return outnames


def crai(session, map_path, outname=None, use_pd=True, n_objects=None, split_pred=True, fit_in_map=True):
    """

    :param session:
    :param map_path:
    :param outname:
    :param test_arg:
    :return:
    """

    t0 = time.time()
    print('Loading data')
    map_id, mrc = get_mrc_from_input(map_path=map_path, session=session)
    mrc = clean_mrc(mrc)
    print(map_id)
    print('Data loaded in : ', time.time() - t0)

    outname = get_outname(outname=outname, map_path=map_path, session=session)
    if outname is None or mrc is None:
        return None
    outnames = predict_coords(mrc=mrc, outname=outname, use_pd=use_pd, n_objects=n_objects, session=session,
                              split_pred=split_pred)
    for outname in outnames:
        ab = run(session, f"open {outname}")
        if fit_in_map:
            run(session, f"fit #{ab[0].id_string} inmap #{map_id}")


crai_desc = CmdDesc(required=[("map_path", StringArg)],
                    optional=[("outname", StringArg),
                              ("use_pd", BoolArg),
                              ("n_objects", IntArg),
                              ("split_pred", BoolArg),
                              ("fit_in_map", BoolArg),
                              ], )
