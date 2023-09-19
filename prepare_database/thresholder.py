import os
import sys

import mrcfile
import numpy as np
from scipy.optimize import root_scalar
import subprocess

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils.mrc_utils import MRCGrid


def r_nz(mrcfile_path, threshold):
    map_data = mrcfile.open(mrcfile_path, mode='r').data
    return np.sum(map_data > threshold) / np.sum(map_data > 0)


def root_scalar_nz(mrcfile_path, target_value=0.445):
    res = root_scalar(lambda t: target_value - r_nz(mrcfile_path, t), bracket=[0, 1])
    return res.root, res.converged


def root_scalar_sa(mrcfile_path, target_value=0.933):
    output = subprocess.run(['/usr/bin/chimerax', '--nogui', '--exit',
                             '--script', f"measure.py {mrcfile_path} {target_value}"],
                            capture_output=True)
    lines = str(output.stdout).split('\\n')
    thresh = None
    steps = 50
    for line in lines:
        if line.startswith('RESULT BRENTS'):
            thresh, steps = line.split()[-2:]
    return float(thresh), int(steps) < 50


def get_thresh(mrcfile_path):
    thresh_sa, converged_sa = root_scalar_sa(mrcfile_path=mrcfile_path)
    thresh_nz, converged_nz = root_scalar_nz(mrcfile_path=mrcfile_path)
    if converged_nz and converged_sa:
        return np.average([thresh_sa, thresh_nz], weights=[0.22, 0.62])
    else:
        return -1


def get_all_nz(mrcfile_path):
    map_data = mrcfile.open(mrcfile_path, mode='r').data
    all_nz = []
    for thresh in np.linspace(0, 1, num=50):
        extracted = map_data[map_data > thresh]
        all_nz.append(np.std(extracted))
        # plt.hist(extracted)
        # plt.show()
        # all_nz.append(np.sum(map_data > thresh) / map_data.size)
        # all_nz.append(np.sum(map_data > thresh) / np.sum(map_data > 0))
    non_zeroes = np.asarray(all_nz)
    return non_zeroes


def get_all_surf_vol(mrcfile_path):
    output = subprocess.run(['/usr/bin/chimerax', '--nogui', '--exit',
                             '--script', f"get_all.py {mrcfile_path}"],
                            capture_output=True)
    lines = str(output.stdout).split('\\n')
    vols, surfs = [], []
    for line in lines:
        if line.startswith('RESULT BRENTS'):
            vol, surf = line.split()[-2:]
            vols.append(float(vol) + 5)
            surfs.append(float(surf) + 5)

    vols = np.asarray(vols)
    surfs = np.asarray(surfs)
    return vols, surfs


if __name__ == '__main__':
    datadir_name = "../data/pdb_em/"
    # dirname = "7V3L_31683"  # 0 - 0.15 looks good. max std 0.18
    # dirname = "7LO8_23464"  # 0.05 - 0.1 looks good. max std 0.10
    dirname = "6NQD_0485"  # 0.06 - 0.14 looks good. max std 0.11

    pdb_name, mrc = dirname.split("_")
    map_path = os.path.join(datadir_name, dirname, f"full_crop_resampled_2.mrc")
    # map_path = os.path.join(datadir_name, dirname, f"emd_{mrc}.map")
    full_cleaned = os.path.join(datadir_name, dirname, f"full_cleaned.mrc")
    thresholded = os.path.join(datadir_name, dirname, f"thresholded.mrc")

    # mrc = MRCGrid.from_mrc(mrc_file=map_path)
    # mrc = mrc.resample().normalize(normalize_mode='max')
    # mrc.save(full_cleaned, overwrite=True)

    import matplotlib.pyplot as plt

    # nzs = get_all_nz(full_cleaned)
    # fig, ax1 = plt.subplots()
    # ax1.plot(np.linspace(0, 1, num=50), nzs, color='blue', label="volume")
    # ax1.set_xlabel('Probability threshold')
    # ax1.set_ylim(0, 0.05)
    # plt.legend()
    # plt.show()

    # vols, surfs = get_all_surf_vol(full_cleaned)
    # fig, ax1 = plt.subplots()
    # ax1.plot(np.linspace(0, 1, num=50), vols, color='blue', label="volume")
    # ax1.plot(np.linspace(0, 1, num=50), surfs, color='red', label="surface")
    # ax1.set_ylim(0, 1e6)
    # plt.legend()
    # ax2 = ax1.twinx()
    # ax2.plot(np.linspace(0, 1, num=50), surfs / (vols + 2), color='green', label="ratio")
    # ax1.set_xlabel('Probability threshold')
    # plt.legend()
    # plt.show()

    import time
    t0 = time.time()
    res = root_scalar_sa(mrcfile_path=full_cleaned)
    print('Density value : ', res, ', time needed : ', time.time() - t0)
    t0 = time.time()
    res = root_scalar_nz(mrcfile_path=full_cleaned)
    print('Density value : ', res, ', time needed : ', time.time() - t0)

    # thresh = get_thresh(mrcfile_path=full_cleaned)
    # print(thresh)
    # thresh=0.005
    #
    # nz = np.sum(mrc.data > thresh) / np.sum(mrc.data > 0)
    # print(nz)
    #
    # mrc.data[mrc.data < thresh] = 0
    # mrc.normalize().save(thresholded, overwrite=True)
