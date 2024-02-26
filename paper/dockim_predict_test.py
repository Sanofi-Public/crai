import os
import sys

import numpy as np
import pickle
import pymol2
import string
from scipy.spatial.transform import Rotation

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from paper.predict_test import get_pdbsels, string_rep

PHENIX_DOCK_IN_MAP = f"{os.environ['HOME']}/bin/phenix-1.20.1-4487/build/bin/phenix.dock_in_map"
UPPERCASE = string.ascii_uppercase
LOWERCASE = string.ascii_lowercase


def get_systems(csv_in='../data/csvs/sorted_filtered_test.csv',
                test_path="../data/testset/",
                nano=False):
    os.makedirs(test_path, exist_ok=True)
    out_name_pdbsels = os.path.join(test_path, f'pdb_sels{"_nano" if nano else ""}.p')
    pdb_selections = get_pdbsels(csv_in=csv_in, out_name=out_name_pdbsels)
    rotated_chain_mapping = {}
    # Get rotated pdbs
    # for dockim comparison, we want to get more copies and also to rename the chains.
    with pymol2.PyMOL() as p:
        p.cmd.feedback("disable", "all", "everything")
        all_pdb_chain_mapping = {}
        for step, ((pdb, mrc, resolution), selections) in enumerate(pdb_selections.items()):
            if not step % 20:
                print(f"Done {step} / {len(pdb_selections)}")
            new_dir_path = os.path.join(test_path, f'{pdb}_{mrc}')
            new_pdb_path = os.path.join(new_dir_path, f"{pdb}.cif")
            p.cmd.load(new_pdb_path, 'in_pdb')
            # Using 10%1 avoids choosing an arbitrary number like 3 copies of A and B and 2 of C and D (10=3+3+2+2)
            # => we get 3 copies of each
            # We cannot afford n_copies*len(selections) to go over 13 for fabs (since we only have 26 uppercase)
            # 5x2, 4x3, 3x4, 4x3, 2x5, 2x6 works but fails for more systems (we don't have the issue)
            n_copies = 10 // (len(selections)) + 10 % 1 != 0
            last_chain = 0
            pdb_chain_mapping = {}
            for i, selection in enumerate(selections):
                sel = f'in_pdb and ({selection})'
                p.cmd.extract(f"to_align", sel)
                original_chains = selection.split(' or ')
                original_chains = [chain.strip('chain ') for chain in original_chains]
                current_chains = original_chains
                coords = p.cmd.get_coords("to_align")
                # print(pdb, selection, n_copies, coords.shape)
                # Make rotated copies of abs
                for k in range(n_copies):
                    rotated = Rotation.random().apply(coords)
                    translated = rotated + np.array([10, 20, 30])[None, :]
                    p.cmd.load_coords(translated, "to_align", state=1)

                    # Handle chain renaming that is important for dockim output
                    # nAbs are not a big problem, but we need to avoid collision for Fabs.
                    # Save in a dict map = {(i,k): (old_chains, new_chains)}
                    if nano:
                        new_chain = LOWERCASE[last_chain]
                        last_chain += 1
                        p.cmd.alter('to_align', f"chain='{new_chain}'")
                        pdb_chain_mapping[(i, k)] = (original_chains, new_chain)
                    else:
                        # We want to go from X,Y => A,B avoiding collisions
                        old_chain_1, old_chain_2 = current_chains
                        new_chain_1, new_chain_2, temp_chain = [UPPERCASE[last_chain + offset] for offset in range(3)]
                        last_chain += 2
                        # Y != A: X,Y -> A,Y -> A,B
                        if not new_chain_1 == old_chain_2:
                            p.cmd.alter(f'to_align and chain {old_chain_1}', f"chain='{new_chain_1}'")
                            p.cmd.alter(f'to_align and chain {old_chain_2}', f"chain='{new_chain_2}'")
                        # We now want X,A => A,B
                        # X,A -> C,A -> C,B -> A,B
                        else:
                            p.cmd.alter(f'to_align and chain {old_chain_1}', f"chain='{temp_chain}'")
                            p.cmd.alter(f'to_align and chain {old_chain_2}', f"chain='{new_chain_2}'")
                            p.cmd.alter(f'to_align and chain {temp_chain}', f"chain='{new_chain_1}'")
                        current_chains = new_chain_1, new_chain_2
                        pdb_chain_mapping[(i, k)] = ((old_chain_1, old_chain_2), (new_chain_1, new_chain_2))
                    outpath_rotated = os.path.join(new_dir_path, f'rotated_{"nano_" if nano else ""}{i}_{k}.pdb')
                    p.cmd.save(outpath_rotated, 'to_align')
                p.cmd.delete("to_align")
            # print(pdb, pdb_chain_mapping)
            all_pdb_chain_mapping[pdb] = pdb_chain_mapping
            p.cmd.delete("in_pdb")
    outname_mapping = os.path.join(test_path, f'dockim_chain_map{"_nano" if nano else ""}.p')
    pickle.dump(all_pdb_chain_mapping, open(outname_mapping, 'wb'))


if __name__ == '__main__':
    # GET DATA
    for sorted_split in [True, False]:
        test_path = f'../data/testset{"" if sorted_split else "_random"}'
        for nano in [False, True]:
            # for nano in [True]:
            csv_in = f'../data/{"nano_" if nano else ""}csvs/{"sorted_" if sorted_split else ""}filtered_test.csv'
            print('Getting data for ', string_rep(sorted_split=sorted_split, nano=nano))
            get_systems(csv_in=csv_in, nano=nano, test_path=test_path)
            # Now let us get the prediction in all cases
            print('Making predictions for :', string_rep(nano=nano))
            # make_predictions(nano=nano, test_path=test_path, use_mixed_model=mixed, gpu=1)
            # get_hit_rates(nano=nano, test_path=test_path, use_mixed_model=mixed)
