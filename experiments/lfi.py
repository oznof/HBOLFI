
import sys
import os
import csv
import copy
import dill
from pathlib import Path
import argparse
import time
from collections import OrderedDict

import torch
from torch.distributions import kl_divergence
import gpytorch
from gpytorch.kernels import AdditiveKernel

from hbolfi import make_gp, BOLFI
from hbolfi import partition_metrics, summary_stats
from hbolfi.simulators import QuadFormModel, SemiparametricQuadFormModel

from .prepare_lfi import prepare_experiment

DEBUG = False

if __name__ == '__main__':
    # ==================================================================================================================
    # Parse arguments
    parser = argparse.ArgumentParser(description="Discrepancy function as proxy simulation model (2D+)")
    parser.add_argument('--simulator', type=str, nargs='?', default="QuadFormModel",
                        help="Simulator name")
    parser.add_argument('--model_type', type=str, nargs='?', default="full", help='Type of model')
    parser.add_argument('--kernel_type', type=str, nargs='?', default="ScaleRBFKernel", help='Type of kernel')
    parser.add_argument('--model_update', type=int, nargs='?', default=10, help="Optimize model every k acquisitions")
    parser.add_argument('--model_iters', type=int, nargs='?', default=50, help="Number of iterations per update")
    parser.add_argument('--problem', choices=['pairs', 'quads', 'halve'], default='pairs', help="type of problem")
    parser.add_argument('--num_dims', type=int, nargs='?', default=4, help="dimension of the problem.")
    parser.add_argument('--path', type=str, nargs='?', default="~/storage/HBOLFI/", help="Results dir.")
    parser.add_argument('--acq_rule', type=str, nargs='?', default="LCB", help="Acquisition rule.")
    parser.add_argument('--acq_noise', type=float, nargs='?', default=1e-4,
                        help="Noise to be added in the acquisition step.")
    parser.add_argument('--seed', type=int, nargs='?', default=117, help="RNG seed")
    parser.add_argument('--corr', type=float, nargs='?', default=0.6, help="Correlation between variables in same block")
    parser.add_argument('--mu_min', type=float, nargs='?', default=0.25, help="Minimum of ground truth function")
    parser.add_argument('--name_suffix', type=str, nargs='?', default='')

    args = parser.parse_args()
    settings = vars(args)
    settings['path'] = settings['path'].replace('~', str(Path.home()))

    settings = prepare_experiment(settings, make_dirs=not DEBUG)
    gpytorch.settings._fast_covar_root_decomposition._state = False
    gpytorch.settings._fast_log_prob._state = False
    gpytorch.settings._fast_solves._state = False
    gpytorch.settings.fast_pred_var._state = False
    gpytorch.settings.fast_pred_samples._state = False
    gpytorch.settings.lazily_evaluate_kernels._state = True
    gpytorch.settings.num_trace_samples._global_value = 0

    SIMULATOR = settings['simulator']
    EXP_PATH = settings['exp_path']
    EXP_NAME = settings['exp_name']
    STATS_PATH = settings['stats_path']
    STATS_UPDATE = settings['stats_update']
    SAVE_MODEL_PATH = settings['save_model_path']
    SAVE_MODEL_UPDATE = settings['save_model_update']
    SEED = settings.pop("seed")
    TRUE_PARTITION = settings['true_partition']
    SEARCH_SPACE = settings['search_space']
    TRUE_KERNEL = settings.get('true_kernel')

    print("Experiment: ", EXP_NAME)
    print("Save path: ", EXP_PATH)
    # print("True kernel:\n", TRUE_KERNEL)
    print("True partition: ",  TRUE_PARTITION)
    # for approximation
    MODEL_TYPE = settings['model_type']
    MODEL_NAME = settings['model_name']
    MODEL_UPDATE = settings['model_update']
    MODEL_ITERS = settings['model_iters']
    MODEL_MODIFIER = settings['model_modifier']
    ACQUISITION = settings['acquisition']
    INITIAL_PARTITION = settings.pop('initial_partition')
    INITIAL_KERNEL = settings.pop('initial_kernel')

    # Ground truth Description
    if SIMULATOR == 'SemiparametricQuadFormModel':
        true_model = SemiparametricQuadFormModel(kernel=TRUE_KERNEL, seed=SEED, **settings)
    elif SIMULATOR == 'QuadFormModel':
        true_model = QuadFormModel(partition=TRUE_PARTITION, seed=SEED, **settings)
    else:
        raise NotImplementedError

    true_loc = true_model.min_loc
    true_value = true_model.min_val
    assert(true_value == 0.)
    threshold = true_value  # + 2 * math.sqrt(settings['noise'])
    bolfi_gtruth = BOLFI(obj_function=true_model.observe, model=true_model,
                         search_space=SEARCH_SPACE, initial_samples=0)
    mvn_lap_ref = bolfi_gtruth.infer(threshold, method='laplace', initial_params=true_loc, skip_map_estimation=True)

    if DEBUG:
        from hbolfi.utils import plot_contour
        import matplotlib.pyplot as plt
        import numpy as np
        orig = lambda x: true_model.observe_noiseless(x.repeat(1, settings['num_groups']))
        plot_contour(orig, fill=True,
                     level_fn=lambda vmin, vmax: np.logspace(np.log10(vmin), np.log10(vmax), 50))
        plt.show()
        orig = lambda x: bolfi_gtruth.unnormalized_loglikelihood(threshold, x.repeat(1, settings['num_groups'])).exp()
        lap = lambda x: mvn_lap_ref.log_prob(x.repeat(1, settings['num_groups'])).exp()
        # ax = plot_contour(orig, minmax_scale=True, fill=True, cmap=cm.gray, levels=np.linspace(0, 1, 50))
        ax = plot_contour(orig, minmax_scale=True, colors='k', linestyles='dashed', linewidths=2)
        plot_contour(lap, colors='r', ax=ax, linewidths=2)
        plt.show()
        sys.exit(1)

    gtruth_file = os.path.join(EXP_PATH, f"{EXP_NAME}_gtruth.pkl")
    if os.path.exists(gtruth_file):
        with open(gtruth_file, "rb") as f:
            obj = dill.load(f)
    else:
        print("Creating test set")
        samples, rhat, ess = bolfi_gtruth.infer(threshold, method='nuts', initial_params=true_loc,
                                                initialize_map=False, seed=SEED, **settings['nuts_options'])
        obj = {"bolfi": bolfi_gtruth, "samples": samples, "ess": ess, "rhat": rhat}
        with open(gtruth_file, "wb") as f:
            dill.dump(obj, f)
    test_inputs = obj["samples"]
    test_mvn = bolfi_gtruth.model(test_inputs)
    test_targets = test_mvn.mean
    test_llik_ref = (bolfi_gtruth.model.likelihood(test_mvn).log_prob(test_targets) /
                     test_targets.size(-1)).item()
    # get top smallest discrepancies
    idx_top = torch.argsort(test_targets)[:100]
    test_inputs_top = test_inputs[idx_top]
    test_targets_top = test_targets[idx_top]
    test_mvn_top = bolfi_gtruth.model(test_inputs_top)
    test_llik_ref_top = (bolfi_gtruth.model.likelihood(test_mvn_top).log_prob(test_targets_top) /
                         test_targets_top.size(-1)).item()
    print("Loaded test set")
    print("Starting BOLFI")
    # Approximation
    if SIMULATOR == 'SemiparametricQuadFormModel':
        mean_function = copy.deepcopy(true_model.mean_function)
    else:
        mean_function = None  # use the default mean (QuadPolyMean) with default priors

    gpm = make_gp(kernel=INITIAL_KERNEL, mean=mean_function)
    bolfi = BOLFI(obj_function=true_model.observe, model=gpm, search_space=SEARCH_SPACE,
                  acquisition=ACQUISITION, seed=SEED)

    if MODEL_MODIFIER:
        selector = MODEL_MODIFIER(model=bolfi.model, seed=SEED)
        MODEL_MODIFIER = lambda: selector.run(MODEL_ITERS)
    for i in range(settings['num_acqs'] + 1):
        start_time = time.time()
        fit = i % MODEL_UPDATE == 0
        bolfi.step(acquire=i > 0, fit=fit, model_modifier=MODEL_MODIFIER if fit else None)
        # Compute stats
        if i % STATS_UPDATE == 0 and fit:
            exp_loc_min, exp_val_min = bolfi.expected_min()
            regret = (true_model.observe_noiseless(exp_loc_min) - true_value).item()
            loc_error = torch.max(torch.abs(exp_loc_min - true_loc)).item()
            res_dict = OrderedDict(zip(("iter", "exp_loc_min", "exp_val_min", "loc_error", "regret"),
                                       (i, exp_loc_min.numpy(), exp_val_min, loc_error, regret)))
            res_dict = summary_stats(bolfi.model, test_inputs, test_targets, test_llik_ref, res_dict)
            res_dict = summary_stats(bolfi.model, test_inputs_top, test_targets_top, test_llik_ref_top,
                                     res_dict, suffix='_top')
            lb, ub = settings['search_space']['bounds']
            near_b = not (torch.all(lb + 0.1 <= exp_loc_min) and torch.all(exp_loc_min <= ub - 0.1))
            kl_lap = float('nan')
            if not near_b:
                mvn_lap = bolfi_gtruth.infer(threshold=exp_val_min, method='laplace', initial_params=exp_loc_min,
                                             skip_map_estimation=True)
                kl_lap = kl_divergence(mvn_lap_ref, mvn_lap).item()

            res_dict.update({'near_b': near_b, 'kl_lap': kl_lap})
            if isinstance(bolfi.model.kernel, AdditiveKernel):
                partition = bolfi.model.additive_structure
            else:
                partition = INITIAL_PARTITION
            res_dict = partition_metrics(TRUE_PARTITION, partition, res_dict=res_dict)
            write_mode = 'a' if i > 0 else 'w'
            with open(os.path.join(STATS_PATH, f"{EXP_NAME}_{MODEL_NAME}_{SEED}.csv"), write_mode) as f:
                w = csv.DictWriter(f, res_dict.keys())
                if i == 0:
                    w.writeheader()
                w.writerow(res_dict)
        if i % SAVE_MODEL_UPDATE == 0:
            with open(os.path.join(SAVE_MODEL_PATH, f"{EXP_NAME}_{MODEL_NAME}_{i}.pkl"), "wb") as f:
                dill.dump({"bolfi": bolfi}, f)
        print(f"Iteration {i:4}  Done. time={time.time() - start_time}s")
