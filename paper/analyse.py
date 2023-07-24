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
def plot_distance(csv_in='../data/csvs/filtered.csv',
                  output_pickle='../data/nano_csvs/benchmark_actual_parsed.p'):
    # get resolution :
    resolutions = get_pdb_selection(csv_in=csv_in, columns=['resolution'])
    dict_res = pickle.load(open(output_pickle, 'rb'))
    dict_res = {pdb: val for (pdb, val) in dict_res.items() if pdb[:4] in resolutions}
    all_resolutions = []
    all_dists_real = []
    failed = 0
    # pdb_selections = get_pdb_selection(csv_in=csv_in, columns=['antibody_selection'])
    for pdb, elt in dict_res.items():
        # if pdb in ['7jvc', '6lht']:
        #     print(pdb, elt)
        if elt is not None:
            all_dists_real.extend(elt[1])
            all_resolutions.extend(resolutions[pdb[:4]])
            # bugs = [x > 12 for x in elt[1]]
            # if any(bugs):
            #     print(pdb, resolutions[pdb.upper()][0][0], [x[0] for x in pdb_selections[pdb.upper()]],
            #           [f"{x:.1f}" for x in elt[1]])
        else:
            failed += 1
            print('failed on : ', pdb)
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


def parse_runtime(output_csv='../data/csvs/benchmark_actual.csv'):
    df_raw = pd.read_csv(output_csv, index_col=0)['dock_runtime']
    runtimes = df_raw.values
    print(sum(runtimes < 0))
    runtimes = runtimes[runtimes > 0]
    print(runtimes.mean())

    plt.hist(np.log10(runtimes), bins=10)
    plt.xlabel("Log10(Time)")
    plt.ylabel("Count")
    plt.show()


if __name__ == '__main__':
    # nano = False
    nano = True
    # sort = False
    sort = True
    print(f'{"nano" if nano else "fab"}, {"sorted" if sort else "random"}')
    if nano:
        # NANO
        csv_in = f'../data/nano_csvs/{"sorted_" if sort else ""}filtered_val.csv'
        output_pickle = '../data/nano_csvs/benchmark_actual_parsed.p'
        output_csv = '../data/nano_csvs/benchmark_actual.csv'
    else:
        csv_in = f'../data/csvs/{"sorted_" if sort else ""}filtered_val.csv'
        # output_file = '../learning/out_big_train_gamma_last.p'
        # output_file = '../learning/out_big_train_gamma_last_old.p'
        # output_file = '../learning/out_big_train_normalize_210.p'
        # output_file = '../learning/out_big_train_normalize_last.p'
        output_pickle = '../data/csvs/benchmark_actual_parsed.p'
        output_csv = '../data/csvs/benchmark_actual.csv'

    # plot_distance(csv_in=csv_in, output_pickle=output_pickle)
    parse_runtime(output_csv=output_csv)
