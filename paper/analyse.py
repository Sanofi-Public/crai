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


def scatter(proba, distances, alpha=0.3, noise_strength=0.02, xlabel='Probability', ylabel='Real Distance', fit=True):
    # Adding random noise to the data

    proba += noise_strength * np.random.randn(len(proba))
    distances += noise_strength * np.random.randn(len(distances))

    # Plotting the scatter data with transparency
    plt.scatter(proba, distances, color='blue', marker='o', alpha=alpha)

    if fit:
        # Linear fit
        m, b = np.polyfit(proba, distances, 1)
        x = np.linspace(proba.min(), proba.max())
        plt.plot(x, m * x + b, color='red')
        # plt.plot(all_probas_bench, m * all_probas_bench + b, color='red', label=f'Linear Fit: y={m:.2f}x+{b:.2f}')

        # Calculating R^2 score
        predicted = m * proba + b
        from sklearn.metrics import r2_score
        r2 = r2_score(distances, predicted)
        plt.text(0.7, 0.9, f'$R^2 = {r2:.2f}$', transform=plt.gca().transAxes)
        plt.text(0.7, 0.85, f'Fit: y={m:.2f}x+{b:.2f}', transform=plt.gca().transAxes)

    # Rest of the plot decorations
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='lower left')
    plt.show()


def final_plot(nano=False, sort=False, average_systems=False, model_name=None, suffix=None):
    # Get our perfomance
    model_name_ref = f"{'n' if nano else 'f'}{'s' if sort else 'r'}_final_last"
    outstring = f"{model_name_ref}_{nano}_{sort}_test.p"
    output_pickle = f"../outfiles/out_{mini_hash(outstring)}_{outstring}"
    dict_res = pickle.load(open(output_pickle, 'rb'))

    if model_name is not None and model_name.startswith('benchmark'):
        # Get benchmark performance : model_name = benchmark_actual_parsed or benchmark_parsed
        benchmark_pickle = f'../data/{"nano_" if nano else ""}csvs/{model_name}.p'
    else:
        if model_name is None:
            model_name = model_name_ref
        outstring = f"{model_name}_{nano}_{sort}_test{suffix if suffix is not None else ''}.p"
        # outstring = f"{model_name}_{nano}_{sort}_test_thresh_pd.p"
        # outstring = f"{model_name}_{nano}_{sort}_test_thresh.p"
        # outstring = f"{model_name}_{nano}_{sort}_test_pd.p"
        benchmark_pickle = f"../outfiles/out_{mini_hash(outstring)}_{outstring}"
    bench_res = pickle.load(open(benchmark_pickle, 'rb'))

    # get resolution :
    csv_in = f'../data/{"nano_" if nano else ""}csvs/{"sorted_" if sort else ""}filtered_test.csv'
    resolutions = get_pdb_selection(csv_in=csv_in, columns=['resolution'])
    resolutions = {pdb: [res.item() for res in resolution] for pdb, resolution in resolutions.items()}

    all_resolutions = []
    all_dists_real = []
    all_dists_real_bench = []
    all_probas_bench = []
    failed = 0
    failed_bench = 0
    for pdb, elt in sorted(dict_res.items()):
        if elt is not None:
            dists = elt[1]
            res = resolutions[pdb[:4]]
            if average_systems:
                all_dists_real.append(np.mean(dists))
                all_resolutions.append(np.mean(res))
            else:
                all_dists_real.extend(dists)
                all_resolutions.extend(res)
        else:
            failed += 1
            # print('failed on : ', pdb)
        if pdb in bench_res and bench_res[pdb] is not None:
            bench_dists = list(bench_res[pdb][1])
            # bench_probas = list(bench_res[pdb][2])
        else:
            bench_dists = []
            bench_probas = []

        # if (any([bench_dist == 20 for bench_dist in bench_dists]) and
        #         all([bench_proba > 0.5 for bench_proba in bench_probas])):
        #     print(pdb, bench_dists, bench_probas)
        #     pass

        if len(bench_dists) == 0:
            failed_bench += 1

        #########################################

        # Complete the list with 20s
        bench_dists = bench_dists + [20 for _ in range(len(elt[1]) - len(bench_dists))]
        if average_systems:
            all_dists_real_bench.append(np.mean(bench_dists))
        else:
            all_dists_real_bench.extend(bench_dists)

        # bench_probas = bench_probas + [0 for _ in range(len(elt[1]) - len(bench_probas))]
        # all_probas_bench.extend(bench_probas)
        # if any([0 < bench_proba < 0.05 for bench_proba in bench_probas]):
        #     print(pdb, bench_dists, bench_probas)
        #     pass
        # print(pdb, elt[1], bench_dists)

    all_dists_real = np.asarray(all_dists_real)
    all_dists_real_bench = np.asarray(all_dists_real_bench)

    # THRESHOLD CAPPING AND COUNTING
    thresh = 10
    # print(thresh)
    # all_dists_real[all_dists_real >= thresh] = np.nan
    # nan = np.nanmean(all_dists_real)
    # count_nan = np.sum(np.isnan(all_dists_real))
    # print(f"{nan:.2f} {count_nan}")
    all_dists_real_bench[all_dists_real_bench >= thresh] = np.nan
    bench_nan = np.nanmean(all_dists_real_bench)
    bench_count_nan = np.sum(np.isnan(all_dists_real_bench))
    print(f"{bench_nan:.2f} {bench_count_nan}")
    print()

    # PROBA VS FAILURE
    # all_probas_bench = np.asarray(all_probas_bench)
    # all_dists_real_bench = np.asarray(all_dists_real_bench)
    # all_dists_real_bench[all_dists_real_bench >= thresh] = thresh
    # all_dists_real_bench = all_dists_real_bench / 6
    # all_dists_real_bench = np.exp(-all_dists_real_bench/6)
    # scatter(all_probas_bench, all_dists_real_bench)
    # plt.scatter(all_probas_bench, all_dists_real_bench)
    # plt.xlabel("Proba")
    # plt.ylabel("Distance")
    # # plt.ylabel("exp(-distance)")
    # plt.show()
    # return

    # # HISTOGRAM PLOTTING
    # uncapped = np.mean(all_dists_real)
    # all_dists_real[all_dists_real >= 20] = 20
    # capped = np.mean(all_dists_real)
    # bench_uncapped = np.mean(all_dists_real_bench)
    # all_dists_real_bench[all_dists_real_bench >= 20] = 20
    # bench_capped = np.mean(all_dists_real_bench)
    # # print(f"Failed on {failed}/{len(dict_res)}, with {len(all_dists_real)} abs")
    # # print(f"Uncapped mean : {uncapped:2f}")
    # # print(f"Capped mean {capped:2f}")
    # # print(f"Bench failed on {failed_bench}/{len(dict_res)}")
    # # print(f"Uncapped mean : {bench_uncapped:2f}")
    # # print(f"Capped mean : {bench_capped:2f}")
    # # print(f"{capped:.2f}/{uncapped:.2f}/{failed} vs "
    # #       f"{bench_capped:.2f}/{bench_uncapped:.2f}/{failed_bench}")
    # plt.rcParams.update({'font.size': 18})
    # # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["cria", "dock in map"])
    # # # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["Ground truth", "Threshold"])
    # # # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["Classic", "Persistence"])
    # # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["Ground truth", "Threshold with PD"])
    # # # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["Threshold with PD", "Threshhold"])
    # plt.hist([all_dists_real, all_dists_real_bench], bins=6, label=["Rz", "Ry"])
    # plt.legend()
    # plt.xlabel("Distance")
    # plt.ylabel("Count")
    # plt.show()

    # all_dists_real[all_dists_real >= 10] = 10
    # plt.scatter(all_resolutions, all_dists_real)
    # plt.xlabel("Resolution")
    # plt.ylabel("Distance")
    # plt.show()
    return {'res': all_resolutions, 'native': all_dists_real, 'bench': all_dists_real_bench}


def compare_bench(average_systems=True):
    for fab in [False, True]:
        for sort in [False, True]:
            first = final_plot(fab, sort, average_systems=average_systems)
            thresh_pd = final_plot(fab, sort, average_systems=average_systems, suffix='_thresh_pd')
            actual = final_plot(fab, sort, average_systems=average_systems, model_name="benchmark_actual_parsed")
            template = final_plot(fab, sort, average_systems=average_systems, model_name="benchmark_parsed")
            # all_sys = [ff, ff_actual, ff_template, ft, ft_actual, ft_template]
            all_sys = [first, thresh_pd, actual, template]
            all_dists_real = [[elt if elt < 20 else 20 for elt in final['bench']] for final in all_sys]
            plt.rcParams.update({'font.size': 18})
            # labels = ['CrIA', 'Dock in map - GT', 'Dock in map - Template']
            labels = ['CrIA', 'CrIA thresh', 'Dock in map - GT', 'Dock in map - Template']
            plt.hist(all_dists_real, bins=6, label=labels)
            plt.legend()
            plt.show()


def do_all(average_systems=False):
    ff = final_plot(False, False, average_systems=average_systems)
    ft = final_plot(False, True, average_systems=average_systems)
    tf = final_plot(True, False, average_systems=average_systems)
    tt = final_plot(True, True, average_systems=average_systems)
    all_sys = [ff, ft, tf, tt]

    # RESOLUTION/PERFORMANCE
    all_resolutions = np.asarray([elt for final in all_sys for elt in final['res']]).flatten()
    all_dists_real = np.asarray([elt for final in all_sys for elt in final['native']]).flatten()
    scatter(all_resolutions, all_dists_real, xlabel='Resolution', ylabel='Distance', fit=True)


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


def compute_ablations():
    for model_name in ["fr_uy_last", "fab_random_normalize_last"]:
        for suffix in [None, '_thresh_pd']:
            for average in [True, False]:
                print(model_name, suffix, average)
                final_plot(False, False, model_name=model_name, suffix=suffix, average_systems=average)
    for model_name in ["fr_final_last"]:
        for suffix in ["_pd", '_thresh']:
            for average in [True, False]:
                print(model_name, suffix, average)
                final_plot(False, False, model_name=model_name, suffix=suffix, average_systems=average)


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
    # final_plot(False, False)
    # final_plot(False, True)
    # final_plot(True, False)
    # final_plot(True, True)

    # do_all()

    # compute_ablations()
    # compare_bench()
