import os
import sys

import numpy as np
import pymol2
import string
from scipy.spatial.transform import Rotation

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from paper.predict_test import get_pdbsels

PHENIX_DOCK_IN_MAP = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.dock_in_map"
UPPERCASE = string.ascii_uppercase
LOWERCASE = string.ascii_lowercase


def get_systems(csv_in='../data/csvs/sorted_filtered_test.csv',
                test_path="../data/testset/",
                nano=False):
    os.makedirs(test_path, exist_ok=True)
    out_name_pdbsels = os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p')
    pdb_selections = get_pdbsels(csv_in=csv_in, out_name=out_name_pdbsels)
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)}")
            # Make new dir
            new_dir_path = os.path.join(test_path, f'{pdb}_{mrc}')
            new_pdb_path = os.path.join(new_dir_path, f"{pdb}.cif")

            # Get gt and rotated pdbs
            p.cmd.load(new_pdb_path, 'in_pdb')
            # for dockim comparison, we want to get more copies
            n_copies = 10 // (len(selections)) + 1
            for i, selection in enumerate(selections):
                sel = f'in_pdb and ({selection})'
                p.cmd.extract(f"to_align", sel)
                coords = p.cmd.get_coords("to_align")
                # print(pdb, selection, n_copies, coords.shape)
                for k in range(n_copies):
                    rotated = Rotation.random().apply(coords)
                    translated = rotated + np.array([10, 20, 30])[None, :]
                    p.cmd.load_coords(translated, "to_align", state=1)
                    outpath_rotated = os.path.join(new_dir_path, f'rotated_{"nano_" if nano else ""}{i}_{k}.pdb')
                    p.cmd.save(outpath_rotated, 'to_align')
                p.cmd.delete("to_align")
            p.cmd.delete("in_pdb")
