import os
import sys

import argparse
import time
import torch
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(script_dir)

from learning.predict_coords import predict_coords
from learning.SimpleUnet import SimpleHalfUnetModel

parser = argparse.ArgumentParser(description='')
parser.add_argument("--input",
                    help='Path to the input files. Should be either a .map or .mrc file, or a directory',
                    # required=True,
                    default="crai/src/data/7LO8_resampled.mrc")
parser.add_argument("--output", default=None,
                    help="Optional : Path to the output."
                         " If nothing is provided, this will create a file named {MRC_FILE}_predicted.pdb."
                         " Ignored when ran on a directory.")
parser.add_argument("--n_objects", type=int, default=None,
                    help="Optional : If the number of antibodies is known, it can be provided here.")
parser.add_argument("--predict_dir", action='store_true', default=False,
                    help="Optional : Use if you want to run on every .map or .mrc files in a certain dir, that should "
                         "be provided as input.")
args = parser.parse_args()

# GET MODEL
model_path = os.path.join(script_dir, 'saved_models/ns_final_last.pth')
model = SimpleHalfUnetModel(classif_nano=True, num_feature_map=32)
model.load_state_dict(torch.load(model_path))

# FILENAMES
if args.predict_dir:
    t0 = time.time()
    for file in tqdm(os.listdir(args.input)):
        if not (file.endswith('.mrc') or file.endswith('.map')):
            continue
        in_mrc = os.path.join(args.input, file)
        output = in_mrc.replace(".mrc", "_predicted.pdb").replace(".map", "_predicted.pdb")
        predict_coords(mrc_path=in_mrc, outname=output, model=model,
                       n_objects=args.n_objects, thresh=0.2, classif_nano=True, use_pd=True)
    print('Whole prediction done in : ', time.time() - t0)
else:
    in_mrc = args.input
    if args.output is None:
        args.output = in_mrc.replace(".mrc", "_predicted.pdb").replace(".map", "_predicted.pdb")
    t0 = time.time()
    predict_coords(mrc_path=in_mrc, outname=args.output, model=model,
                   n_objects=args.n_objects, thresh=0.2, classif_nano=True, use_pd=True)
    print('Whole prediction done in : ', time.time() - t0)
