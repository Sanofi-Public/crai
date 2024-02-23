import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from prepare_database.process_data import get_pdb_selection
from utils.python_utils import mini_hash


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


def parse_dict_res(main_dict, keys=('real_dists',), bench_dict=None, actual_benchmark=False, average_systems=False,
                   default_missing_value=20, nano=False, sort=False):
    """
    Function to handle going from the dict created in relog to ones with a list of outputs for each keys.
    For instance : real_dist : [2, 4, 20... ]

    We need two dicts as one is merely a reference of how many abs are in each systems.
    :param main_dict:
    :param keys:
    :param bench_dict:
    :param actual_benchmark:
    :param average_systems:
    :param default_missing_value:
    :param nano:
    :param sort:
    :return:
    """
    # get resolution :
    csv_in = f'../data/{"nano_" if nano else ""}csvs/{"sorted_" if sort else ""}filtered_test.csv'
    resolutions = get_pdb_selection(csv_in=csv_in, columns=['resolution'])
    resolutions = {pdb: [res.item() for res in resolution] for pdb, resolution in resolutions.items()}
    all_resolutions = []
    all_pdb_ids = []

    main_results = {k: [] for k in keys}
    bench_results = {k: [] for k in keys}
    if average_systems:
        main_results["raw"] = []
        bench_results["raw"] = []

    per_pdb_results = {}
    failed_bench = 0
    for pdb, metrics in sorted(main_dict.items()):
        n_abs = len(metrics['real_dists'])

        # Try removing an outlier from the computation
        if n_abs > 10:
            continue

        res = resolutions[pdb[:4]]
        if average_systems:
            all_pdb_ids.append(pdb)
            all_resolutions.append(np.mean(res))
        else:
            all_pdb_ids.extend([pdb for _ in range(len(res))])
            all_resolutions.extend(res)
        for k in keys:
            metric_value = metrics[k]
            if average_systems:
                main_results[k].append(np.mean(metric_value))
            else:
                main_results[k].extend(metric_value)

        if bench_dict is not None:
            temp_res = {k: [] for k in keys}
            # Analysis of dock in map, then the result is pdb [dist, real_dist, col_ind]
            if actual_benchmark:
                # You cannot ask for more than distances to benchmark
                assert "real_dists" in keys and len(keys) == 1
                if pdb in bench_dict and bench_dict[pdb] is not None:
                    bench_dists = bench_dict[pdb][1]
                    temp_res["real_dists"] = list(bench_dists)
            # This is just if we are parsing another dict.
            else:
                if pdb in bench_dict:
                    dists = bench_dict[pdb]['real_dists']
                    if dists is not None and len(dists) > 0:
                        for k in keys:
                            temp_res[k] = list(bench_dict[pdb][k])
            if len(next(iter(temp_res.values()))) == 0:
                failed_bench += 1
            for k in keys:
                underprediction = max(n_abs - len(temp_res[k]), 0)
                completed_capped = ([min(default_missing_value, val) for val in temp_res[k]] +
                                    [default_missing_value for _ in range(underprediction)])
                temp_res[k] = completed_capped
                # JUST USEFUL TO FIND USECASES
                # if k=='real_dists':
                #     dists = temp_res[k]
                #     if (
                #         any([dist < 10 for dist in dists])
                #         and any([dist > 10 for dist in dists])
                #         and res[0] > 3
                #     ):
                #         print(pdb, res, dists)
                per_pdb_results[pdb] = temp_res[k]
                if average_systems:
                    if k == 'real_dists':
                        bench_results["raw"].append(temp_res[k])
                    bench_results[k].append(np.mean(temp_res[k]))
                else:
                    bench_results[k].extend(temp_res[k])

    # To get failed systems
    for pdb, values in sorted(per_pdb_results.items()):
        if any([value > 10 for value in values]):
            # print(pdb, values)
            pass
    all_resolutions = np.asarray(all_resolutions)
    for k in keys:
        main_results[k] = np.asarray(main_results[k])
        bench_results[k] = np.asarray(bench_results[k])
    # print("Overpredictions : ", overpredictions)
    return {'res': all_resolutions, 'native': main_results, 'bench': bench_results, 'pdbs': all_pdb_ids}


def get_results(nano=False, sort=False, average_systems=False, model_name=None, suffix='_pd', keys=('real_dists',)):
    """
    From a set of high level key words, open the right files and parse them

    :param nano:
    :param sort:
    :param average_systems:
    :param model_name:
    :param suffix:
    :return:
    """
    # Get our perfomance
    model_name_ref = f"{'n' if nano else 'f'}{'s' if sort else 'r'}_final_last"
    outstring = f"{model_name_ref}_{nano}_{sort}_test_pd.p"
    output_pickle = f"../outfiles/out_{mini_hash(outstring)}_{outstring}"
    dict_res = pickle.load(open(output_pickle, 'rb'))

    # GET BENCHMARK PERFORMANCE
    if model_name is not None and model_name.startswith('benchmark'):
        actual_benchmark = True
        # Get benchmark performance : model_name = benchmark_actual_parsed or benchmark_parsed
        benchmark_pickle = f'../data/{"nano_" if nano else ""}csvs/{model_name}.p'
    else:
        actual_benchmark = False
        if model_name is None:
            model_name = model_name_ref
        outstring = f"{model_name}_{nano}_{sort}_test{suffix if suffix is not None else '_pd'}.p"
        benchmark_pickle = f"../outfiles/out_{mini_hash(outstring)}_{outstring}"
        # To use the mixed models for Fabs too.
        # model_name = f"n{'s' if sort else 'r'}_final_last"
        # outstring = f"{model_name}_{nano}_{sort}_test{suffix if suffix is not None else '_pd'}.p"
        # benchmark_pickle = f"../outfiles/new_out_{mini_hash(outstring)}_{outstring}"
    bench_res = pickle.load(open(benchmark_pickle, 'rb'))

    all_results = parse_dict_res(main_dict=dict_res, bench_dict=bench_res, keys=keys, actual_benchmark=actual_benchmark,
                                 average_systems=average_systems, nano=nano, sort=sort)
    return all_results


def get_hit_rate(res_dict, average_systems=False, thresh=10):
    """
    Parse a res_dict to compute HR metric. This requires care in the 'average system' settings

    :param res_dict:
    :param average_systems:
    :param thresh:
    :return:
    """
    ref_data = res_dict['native']
    bench_data = res_dict['bench']
    if not average_systems:
        all_dists_real_bench = bench_data['real_dists']
        all_dists_real_bench[all_dists_real_bench >= thresh] = np.nan
        bench_dist = np.nanmean(all_dists_real_bench)
        bench_fails = np.sum(np.isnan(all_dists_real_bench))
        print(f"Dists : {bench_dist:.3f}, "
              f"fails : {bench_fails}, "
              f"HR : {100 * (1 - bench_fails / len(ref_data['real_dists'])):.2f}")
    else:
        all_hr = []
        for system_res in bench_data['raw']:
            hr = 100 * (sum([x < thresh for x in system_res]) / len(system_res))
            all_hr.append(hr)
        # Distance is not meaningful for systems : if one system's mean increases, and go beyond 20, score decreases...
        print(f"{np.mean(all_hr):.3f}")


def compute_hr(nano=False, sort=False, average_systems=False, model_name=None, suffix='_pd', thresh=10):
    """
    Merely a wrapper that combines get_results and get_hr
    :param nano:
    :param sort:
    :param average_systems:
    :param model_name:
    :param suffix:
    :param thresh:
    :return:
    """
    all_results = get_results(nano=nano, sort=sort, average_systems=average_systems, model_name=model_name,
                              suffix=suffix)
    get_hit_rate(all_results, average_systems=average_systems, thresh=thresh)

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
    return all_results


def compute_all(average_systems=True, suffix='_pd', model_name=None):
    ff = compute_hr(False, False, average_systems=average_systems, suffix=suffix, model_name=model_name)
    ft = compute_hr(False, True, average_systems=average_systems, suffix=suffix, model_name=model_name)
    tf = compute_hr(True, False, average_systems=average_systems, suffix=suffix, model_name=model_name)
    tt = compute_hr(True, True, average_systems=average_systems, suffix=suffix, model_name=model_name)


def compare_bench(average_systems=True):
    for fab in [False, True]:
        for sort in [False, True]:
            first = get_results(fab, sort, average_systems=average_systems, suffix='_pd')
            thresh_pd = get_results(fab, sort, average_systems=average_systems, suffix='_thresh_pd')
            actual = get_results(fab, sort, average_systems=average_systems, model_name="benchmark_actual_parsed")
            template = get_results(fab, sort, average_systems=average_systems, model_name="benchmark_parsed")
            # all_sys = [ff, ff_actual, ff_template, ft, ft_actual, ft_template]
            all_sys = [first, thresh_pd, actual, template]
            all_dists_real = [[elt if elt < 20 else 20 for elt in final['bench']['real_dists']] for final in all_sys]

            plt.rcParams.update({'font.size': 14})
            plt.rcParams['text.usetex'] = True
            labels = [r'\texttt{CrAI}',
                      r'\texttt{CrAI thresh}',
                      r'\texttt{dock\_in\_map} GT',
                      r'\texttt{dock\_in\_map} Template',
                      ]
            # labels = ['CrIA', 'Dock in map - GT', 'Dock in map - Template']
            plt.rc('grid', color='grey', alpha=0.5)
            plt.grid(True)

            plt.hist(all_dists_real, bins=6, label=labels)
            plt.legend()
            ax = plt.gca()
            ax.set_xlabel(r'Distance (\AA{})')
            ax.set_ylabel('Count')
            plt.savefig(f'../fig_paper/python/{"fab" if fab else "nano"}_{"sorted" if sort else "random"}.pdf')
            plt.show()


def scatter(proba, distances, alpha=0.3, noise_strength=0.02, xlabel='Probability', ylabel='Real Distance', fit=True):
    # Adding random noise to the data

    proba += noise_strength * np.random.randn(len(proba))
    distances += noise_strength * np.random.randn(len(distances))

    # Rest of the plot decorations
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['text.usetex'] = True
    # a= r'\texttt{dock\_in\_map} Template'
    plt.rc('grid', color='grey', alpha=0.2)
    plt.grid(True)
    # Plotting the scatter data with transparency
    sns_colors = sns.color_palette('colorblind', as_cmap=True)
    plt.scatter(proba, distances, color=sns_colors[0], marker='o', alpha=alpha)

    if fit:
        # Linear fit
        m, b = np.polyfit(proba, distances, 1)
        x = np.linspace(proba.min(), proba.max())
        plt.plot(x, m * x + b, color=sns_colors[1])
        # plt.plot(all_probas_bench, m * all_probas_bench + b, color='red', label=f'Linear Fit: y={m:.2f}x+{b:.2f}')

        # Calculating R^2 score
        predicted = m * proba + b
        from sklearn.metrics import r2_score
        r2 = r2_score(distances, predicted)
        plt.text(0.68, 0.86, rf'$y = {m:.2f} x + {b:.2f}$', transform=plt.gca().transAxes)
        plt.text(0.66, 0.8, rf'$R^2 = {r2:.2f}$', transform=plt.gca().transAxes)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(loc='lower left')
    plt.savefig(f'../fig_paper/python/resolution.pdf')
    plt.show()


def resolution_plot(average_systems=True):
    suffix = '_thresh_pd'
    # model_name = "benchmark_actual_parsed"
    model_name = None
    ff = get_results(False, False, model_name=model_name, suffix=suffix, average_systems=average_systems)
    ft = get_results(False, True, model_name=model_name, suffix=suffix, average_systems=average_systems)
    tf = get_results(True, False, model_name=model_name, suffix=suffix, average_systems=average_systems)
    tt = get_results(True, True, model_name=model_name, suffix=suffix, average_systems=average_systems)
    all_sys = [ff, ft, tf, tt]
    # all_sys = [ft]

    # RESOLUTION/PERFORMANCE
    if average_systems:
        all_dists_real = [individual if individual < 10 else 10
                          for systems in all_sys
                          for system in systems['bench']['raw']
                          for individual in system]
        all_resolutions = [resolution
                           for systems in all_sys
                           for system, resolution in zip(systems['bench']['raw'], systems['res'])
                           for _ in system]
        print([res for res in all_resolutions if res > 5])
        all_dists_real = np.asarray(all_dists_real).flatten()
    else:
        all_resolutions = np.asarray([elt for final in all_sys for elt in final['res']]).flatten()
        print([res for res in all_resolutions if res > 5])
        all_dists_real = [elt if elt < 10 else 10 for final in all_sys for elt in final['bench']['raw']]
        all_dists_real = np.asarray(all_dists_real).flatten()
    scatter(all_resolutions, all_dists_real, xlabel=r'Resolution (\AA{})', ylabel=r'Distance (\AA{})', fit=True)

    suffix = '_thresh_pd'
    model_name = "benchmark_actual_parsed"
    # model_name = None
    ff = get_results(False, False, model_name=model_name, suffix=suffix, average_systems=average_systems)
    ft = get_results(False, True, model_name=model_name, suffix=suffix, average_systems=average_systems)
    tf = get_results(True, False, model_name=model_name, suffix=suffix, average_systems=average_systems)
    tt = get_results(True, True, model_name=model_name, suffix=suffix, average_systems=average_systems)
    all_sys = [ff, ft, tf, tt]
    # all_sys = [ft]

    # RESOLUTION/PERFORMANCE
    if average_systems:
        all_dists_real = [individual if individual < 10 else 10
                          for systems in all_sys
                          for system in systems['bench']['raw']
                          for individual in system]
        all_resolutions = [resolution
                           for systems in all_sys
                           for system, resolution in zip(systems['bench']['raw'], systems['res'])
                           for _ in system]
        print([res for res in all_resolutions if res > 5])
        all_dists_real = np.asarray(all_dists_real).flatten()
    else:
        all_resolutions = np.asarray([elt for final in all_sys for elt in final['res']]).flatten()
        print([res for res in all_resolutions if res > 5])
        all_dists_real = [elt if elt < 10 else 10 for final in all_sys for elt in final['bench']['raw']]
        all_dists_real = np.asarray(all_dists_real).flatten()

    scatter(all_resolutions, all_dists_real, xlabel=r'Resolution (\AA{})', ylabel=r'Distance (\AA{})', fit=True)


def compute_ablations():
    for model_name in ["fr_uy_last", "fab_random_normalize_last"]:
        for suffix in [None, '_thresh_pd']:
            for average in [True, False]:
                print(model_name, suffix, average)
                compute_hr(False, False, model_name=model_name, suffix=suffix, average_systems=average)
    for model_name in ["fr_final_last"]:
        for suffix in ["", '_thresh']:
            for average in [True, False]:
                print(model_name, suffix, average)
                compute_hr(False, False, model_name=model_name, suffix=suffix, average_systems=average)


def get_angles():
    keys = ('real_dists', 'rz_angle', 'theta_angle',)
    # keys = ('real_dists', 'rz_angle', 'rz_norm', 'theta_angle', 'theta_norm',)
    res_fab = get_results(False, False, suffix='_thresh_pd', keys=keys)
    res_nano = get_results(True, False, suffix='_thresh_pd', keys=keys)
    # res_uy = get_results(False, False, suffix='_thresh_pd', keys=keys, model_name='fr_uy_290')

    # Rest of the plot decorations
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['text.usetex'] = True
    plt.rc('grid', color='grey', alpha=0.2)

    # results = [res_fab, res_nano, res_uy]
    # labels = ['Fab', 'nAb', '$\overrightarrow{u_y}$']

    results = [res_fab, res_nano]
    labels = ['Fab', 'nAb']

    # savenames = ['angle_fab', 'angle_nab', 'angle_u_y']
    # for result, label, savename in zip(results, labels, savenames):
    #     extractor = result['bench']['real_dists'] < 10
    #     angles = result['bench']['rz_angle'][extractor] * 180 / 3.14
    #     for key, arr in result['bench'].items():
    #         masked = arr[extractor]
    #         print(key, np.mean(masked) * 180 / 3.14)
    #
    #     plt.hist(angles, bins=np.linspace(0, 180, 19))
    #     plt.grid(True)
    #     plt.title(rf'Result for the {label} model')
    #     plt.xlabel(rf'Angle difference ($^\circ$)')
    #     plt.ylabel(rf'Count')
    #     plt.savefig(f'../fig_paper/python/{savename}.pdf')
    #     plt.show()

    bins = np.linspace(0, 180, 10)
    all_angles = []
    all_names = []
    for result, label in zip(results, labels):
        extractor = result['bench']['real_dists'] < 10
        angles = result['bench']['rz_angle'][extractor] * 180 / 3.14
        for key, arr in result['bench'].items():
            masked = arr[extractor]
            print(key, np.mean(masked) * 180 / 3.14)
        all_angles.extend(angles)
        all_names.extend([label for _ in range(len(angles))])

    df = pd.DataFrame({'angles': all_angles,
                       'Model Name': all_names})
    hist_kwargs = {
        "multiple": "dodge",
        "stat": "percent",
        "kde": False,
        "bins": bins,
        "common_norm": False,
    }
    sns.histplot(data=df, x='angles', hue='Model Name', shrink=.9, **hist_kwargs)
    # sns.histplot(all_angles[0], label=labels[0], **hist_kwargs)
    # sns.histplot(all_angles[1], label=labels[1], **hist_kwargs)
    # sns.histplot(all_angles[2], label=labels[2], **hist_kwargs)
    # plt.hist(all_angles[0], label=labels[0], **hist_kwargs)
    # plt.hist(all_angles[1], label=labels[1], **hist_kwargs)
    # plt.hist(all_angles[2], label=labels[2], **hist_kwargs)
    # bw_method = 1
    # sns.kdeplot(all_angles[0], label=labels[0], clip=(0, 180), bw_method=bw_method)
    # sns.kdeplot(all_angles[1], label=labels[1], clip=(0, 180), bw_method=bw_method)
    # sns.kdeplot(all_angles[2], label=labels[2], clip=(0, 180), bw_method=bw_method)
    # sns.histplot(all_angles, kde=True, label=labels, bins=np.linspace(0, 180, 19))
    plt.grid(True)
    plt.xlabel(rf'Angle difference ($^\circ$)')
    plt.ylabel(rf'Percent')
    plt.savefig(f'../fig_paper/python/angles.pdf')
    plt.show()


if __name__ == '__main__':
    pass
    # output_csv = '../data/csvs/benchmark_actual.csv'
    # parse_runtime(output_csv=output_csv)

    # # SELECT 8GOO_34178, resolution 4.4 as successful prediction
    # # SELECT 8CXI_27058 , resolution 3.4 as partial success
    # # SELECT 7Z85_14543, resolution 3.1 as nano success
    sort = True
    nano = True
    # suffix = '_thresh_pd'
    # ours = get_results(nano=nano, sort=sort, suffix=suffix, average_systems=True)
    # bench = get_results(nano=nano, sort=sort, model_name="benchmark_actual_parsed", average_systems=True)
    # pdbs = ours['pdbs']
    # res = ours['res']
    # plt.hist(res)
    # plt.show()
    # print(np.mean(res))
    # print(res)
    # ours = ours['bench']['raw']
    # bench = bench['bench']['raw']
    # Buggy one with 16 fabs is an outlier :
    # print(sorted([len(x) for x in bench]))
    # argsort = np.argsort(res)
    # with open('chiara_fab.txt', 'w') as f:
    #     for i in argsort:
    #         # if float(res[i])<5: continue
    #         line = "Resolution :" + str(res[i]) + " id :" + str(pdbs[i]) + \
    #                " ours : " + str(ours[i]) + " bench :" + str(bench[i]) + "\n"
    #         # if str(pdbs[i]) == "7YAI_33713":
    #         #     print("yes")
    #         print(line.strip())
    #         # f.writelines(line)
    # a = 1

    model_name = None
    # model_name = "benchmark_actual_parsed"
    average_systems = False
    # average_systems = False
    # suffix = ''
    # suffix = '_pd'
    suffix = '_thresh_pd'
    compute_all(model_name=model_name, average_systems=average_systems, suffix=suffix)

    # compare_bench()

    # resolution_plot()

    # compute_ablations()

    # get_angles()

# This is on Fabs
# Failed Fabs
# 7XDA_33140
# 7XJ6_33220
# 7YVN_34133
# 7YVO_34134
# 7YVP_34135
# 8CXI_27058
# 8D0Z_27113
# 8DEF_27392
# 8DWW_27763
# 8DWX_27765
# 8DWY_27767
# 8E8L_27943
# 8E8R_27947
# 8E8X_27949
# 8H07_34410
# 8HEC_34687

# Failed Nanos
# 7YC5_33734
# 7YM8_33924
# 7YMH_33928
# 8EQB_28536
# 8G8W_29857
# 8GNI_34165
# 8GNJ_34166
# 8GQ5_34198
# 8GW8_34305
# 8HMP_34902
# 8J1N_35928
