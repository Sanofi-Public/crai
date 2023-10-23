from chimerax.core.commands import CmdDesc
from chimerax.core.commands import StringArg
from chimerax.core.commands import run

import os
import sys

import time
import numpy as np
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir))

import mrc_utils
from SimpleUnet import SimpleHalfUnetModel
from object_detection import output_to_transforms, transforms_to_pdb_biopython


def crop_large_mrc(mrc, margin=12):
    arr = mrc.data
    to_find = arr > 0.1
    res = np.nonzero(to_find)
    all_min_max = []
    for r, shape in zip(res, to_find.shape):
        min_i, max_i = np.min(r), np.max(r)
        min_i, max_i = max(0, min_i - margin), min(max_i + margin, shape)
        all_min_max.append(min_i)
        # The max_i is computed from the end
        all_min_max.append(shape - max_i)
    return all_min_max


def predict_coords(mrc_path, resample=True, normalize='max', outname=None, outmrc=None,
                   n_objects=None, thresh=0.5, crop=0, classif_nano=False, default_nano=False, use_pd=False):
    t0 = time.time()
    print('Loading data')
    mrc = mrc_utils.MRCGrid.from_mrc(mrc_path)
    if resample:
        mrc = mrc.resample()
    mrc = mrc.normalize(normalize_mode=normalize)
    if crop != 0:
        mrc = mrc.crop(*(crop,) * 6)
    else:
        all_min_max = crop_large_mrc(mrc)
        mrc = mrc.crop(*all_min_max)
    mrc_grid = torch.from_numpy(mrc.data[None, None, ...])
    print('Data loaded in : ', time.time() - t0)

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
                                      classif_nano=classif_nano, default_nano=default_nano, use_pd=use_pd)
    transforms_to_pdb_biopython(transforms=transforms, out_name=outname)
    print('Done ! Output saved in ', outname)
    return outname


def crai(session, map_path, outname=None):
    if outname is None:
        default_outname = map_path.replace(".mrc", "_predicted.pdb").replace(".map", "_predicted.pdb")
        if not os.path.exists(default_outname):
            outname = default_outname
        else:
            print("Default name not available, one could not save the prediction. Please add an outname.")
            return None
    outname = predict_coords(map_path, outname=outname)
    run(session, f"open {outname}")


crai_desc = CmdDesc(required=[("map_path", StringArg)],
                    optional=[("outname", StringArg)])
