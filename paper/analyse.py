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
from utils.python_utils import mini_hash


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

    # plt.hist(all_dists_real, bins=10)
    # plt.xlabel("Distance")
    # plt.ylabel("Count")
    # plt.show()

    plt.scatter(all_resolutions, all_dists_real)
    plt.xlabel("Resolution")
    plt.ylabel("Distance")


def final_plot(nano=False, sort=False):
    # Get our perfomance
    model_name = f"{'n' if nano else 'f'}{'s' if sort else 'r'}_final_last"
    outstring = f"{model_name}_{nano}_{sort}_test.p"
    output_pickle = f"../outfiles/out_{mini_hash(outstring)}_{outstring}"
    dict_res = pickle.load(open(output_pickle, 'rb'))

    # Get benchmark performance
    # benchmark_pickle = f'../data/{"nano_" if nano else ""}csvs/benchmark_actual_parsed.p'
    # outstring = f"{model_name}_{nano}_{sort}_test.p"
    # outstring = f"{model_name}_{nano}_{sort}_test_thresh.p"
    # outstring = f"{model_name}_{nano}_{sort}_test_pd.p"
    outstring = f"{model_name}_{nano}_{sort}_test_thresh_pd.p"
    # benchmark_pickle = f"../outfiles/old_ones/out_{mini_hash(outstring)}_{outstring}"
    benchmark_pickle = f"../outfiles/out_{mini_hash(outstring)}_{outstring}"
    bench_res = pickle.load(open(benchmark_pickle, 'rb'))

    # get resolution :
    csv_in = f'../data/{"nano_" if nano else ""}csvs/{"sorted_" if sort else ""}filtered_test.csv'
    resolutions = get_pdb_selection(csv_in=csv_in, columns=['resolution'])

    #
    all_resolutions = []
    all_dists_real = []
    all_dists_real_bench = []
    failed = 0
    failed_bench = 0
    for pdb, elt in sorted(dict_res.items()):
        if elt is not None:
            pass
            all_dists_real.extend(elt[1])
            # all_dists_real.append(np.mean(elt[1]))
            # all_resolutions.extend(resolutions[pdb[:4]])
        else:
            failed += 1
            # print('failed on : ', pdb)
        if pdb in bench_res and bench_res[pdb] is not None:
            bench_dists = list(bench_res[pdb][1])
        else:
            bench_dists = []
        if len(bench_dists) == 0:
            failed_bench += 1
            # print('failed on : ', pdb)
        # Complete the list with 20s
        bench_dists = bench_dists + [20 for _ in range(len(elt[1]) - len(bench_dists))]
        all_dists_real_bench.append(np.mean(bench_dists))
        # all_dists_real_bench.extend(bench_dists)
        # print(pdb, elt[1], bench_dists)
    all_dists_real = np.asarray(all_dists_real)
    uncapped = np.mean(all_dists_real)
    # all_dists_real[all_dists_real >= 20] = 20
    capped = np.mean(all_dists_real)

    thresh = 10
    # print(thresh)
    all_dists_real[all_dists_real >= thresh] = np.nan
    nan = np.nanmean(all_dists_real)
    count_nan = np.sum(np.isnan(all_dists_real))
    # print(f"Failed on {failed}/{len(dict_res)}")
    # print(f"Uncapped mean : {uncapped:2f}")
    # print(f"Capped mean {capped:2f}")
    # print(len(all_dists_real))
    all_dists_real_bench = np.asarray(all_dists_real_bench)
    bench_uncapped = np.mean(all_dists_real_bench)
    # all_dists_real_bench[all_dists_real_bench >= 20] = 20
    bench_capped = np.mean(all_dists_real_bench)
    all_dists_real_bench[all_dists_real_bench >= thresh] = np.nan
    bench_nan = np.nanmean(all_dists_real_bench)
    bench_count_nan = np.sum(np.isnan(all_dists_real_bench))
    # print(f"Bench failed on {failed_bench}/{len(dict_res)}")
    # print(f"Uncapped mean : {bench_uncapped:2f}")
    # print(f"Capped mean : {bench_capped:2f}")
    # print(f"{capped:.2f}/{uncapped:.2f}/{failed} vs "
    #       f"{bench_capped:.2f}/{bench_uncapped:.2f}/{failed_bench}")
    # print(f"{nan:.2f} {count_nan} {bench_nan:.2f} {bench_count_nan}")
    print(f"{bench_nan:.2f} {bench_count_nan}")
    # print()

    plt.rcParams.update({'font.size': 18})
    # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["cria", "dock in map"])
    # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["Ground truth", "Threshold"])
    # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["Classic", "Persistence"])
    plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["Ground truth", "Threshold with PD"])
    # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["Threshold with PD", "Threshhold"])
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("Count")
    # plt.show()

    # plt.scatter(all_resolutions, all_dists_real)
    # plt.xlabel("Resolution")
    # plt.ylabel("Distance")


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
    nano = False
    # nano = True
    sort = False
    # sort = True
    # print(f'{"nano" if nano else "fab"}, {"sorted" if sort else "random"}')
    if nano:
        # NANO
        # csv_in = f'../data/nano_csvs/{"sorted_" if sort else ""}filtered_val.csv'
        # csv_in = f'../data/nano_csvs/{"sorted_" if sort else ""}filtered.csv' # nano whole : 22.9, 6.3
        csv_in = f'../data/nano_csvs/{"sorted_" if sort else ""}filtered_test.csv'
        output_csv = '../data/nano_csvs/benchmark_actual.csv'
        output_pickle = '../data/nano_csvs/benchmark_actual_parsed.p'
    else:
        # csv_in = f'../data/csvs/{"sorted_" if sort else ""}filtered_val.csv'
        # csv_in = f'../data/csvs/{"sorted_" if sort else ""}filtered.csv' # Fab whole : 24.4 , 7.0
        csv_in = f'../data/csvs/{"sorted_" if sort else ""}filtered_test.csv'
        output_csv = '../data/csvs/benchmark_actual.csv'
        # output_pickle = '../outfiles/out_big_train_gamma_last.p'
        # output_pickle = '../outfiles/out_big_train_gamma_last_old.p'
        # output_pickle = '../outfiles/out_big_train_normalize_210.p'
        output_pickle = '../outfiles/out_big_train_normalize_last.p'
        # output_pickle = '../data/csvs/benchmark_actual_parsed.p'

    # plot_distance(csv_in=csv_in, output_pickle=output_pickle)
    # parse_runtime(output_csv=output_csv)
    final_plot(False, False)
    final_plot(False, True)
    final_plot(True, False)
    final_plot(True, True)
