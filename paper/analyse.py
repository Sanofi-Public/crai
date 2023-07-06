import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from prepare_database.process_data import get_pdb_selection


# chunked = pd.read_csv('data/csvs/chunked.csv', index_col=0).reset_index(drop=True)
# chunked.to_csv('chunked.csv')

# Get plots for the extensive validation of a model
def plot_distance():
    # get resolution :
    csv_in = '../data/csvs/filtered.csv'
    resolutions = get_pdb_selection(csv_in=csv_in, columns=['resolution'])

    # output_file = '../learning/out_big_train_gamma_last.p'
    # output_file = '../learning/out_big_train_gamma_last_old.p'
    # output_file = '../learning/out_big_train_normalize_210.p'
    output_file = '../learning/out_big_train_normalize_last.p'
    dict_res = pickle.load(open(output_file, 'rb'))
    dict_res = {pdb_em[:4].lower(): val for pdb_em, val in dict_res.items()}

    # val_systems = set(dict_res.keys())
    # output_file = '../data/csvs/benchmark_actual_parsed.p'
    # dict_res = pickle.load(open(output_file, 'rb'))
    # dict_res = {pdb: val for (pdb, val) in dict_res.items() if pdb in val_systems}

    all_resolutions = []
    all_dists_real = []
    failed = 0
    for pdb, elt in dict_res.items():
        # if pdb in ['7jvc', '6lht']:
        #     print(pdb, elt)
        if elt is not None:
            all_dists_real.extend(elt[1])
            all_resolutions.extend(resolutions[pdb.upper()])
        else:
            failed += 1
            print(pdb)
    print(f"Failed on {failed}/{len(dict_res)}")
    all_dists_real = np.asarray(all_dists_real)

    def hr(distances):
        print(f"Hits at zero distance : {sum(distances <= 0)}/{len(distances)}")
        print(f"Hits below one distance : {sum(distances <= 1)}/{len(distances)}")
        print(f"Hits below six distance : {sum(distances <= 6)}/{len(distances)}")

    hr(all_dists_real)

    plt.rcParams.update({'font.size': 18})
    all_dists_real = np.asarray(all_dists_real)
    print("Uncapped mean : ", np.mean(all_dists_real))
    all_dists_real[all_dists_real >= 20] = 20
    print("Capped mean : ", np.mean(all_dists_real))

    plt.hist(all_dists_real, bins=10)
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.show()

    # plt.scatter(all_resolutions, all_dists_real)
    # plt.xlabel("Resolution")
    # plt.ylabel("Distance")
    # plt.show()


def parse_runtime():
    output_file = '../data/csvs/benchmark_actual.csv'
    df_raw = pd.read_csv(output_file, index_col=0)['dock_runtime']
    runtimes = df_raw.values
    print(sum(runtimes < 0))
    runtimes = runtimes[runtimes > 0]
    print(runtimes.mean())

    plt.hist(np.log10(runtimes), bins=10)
    plt.xlabel("Log10(Time)")
    plt.ylabel("Count")
    plt.show()


if __name__ == '__main__':
    plot_distance()
    # parse_runtime()
