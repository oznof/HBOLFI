import os
import csv
import copy
import dill
import argparse
import time
from collections import OrderedDict

import torch
from torch.distributions import kl_divergence
import gpytorch

from hbolfi import AdditiveStructureGPModel, ExactGPModel, BOLFI
from hbolfi import partition_metrics, summary_stats
from simulators import QuadFormModel, SemiparametricQuadFormModel

from experiment_gp import prepare_experiment, infer_distributions

DEBUG = False

if __name__ == '__main__':
    # ==================================================================================================================
    # Parse arguments
    parser = argparse.ArgumentParser(description="Discrepancy function as proxy simulation model (2D+)")
    parser.add_argument('--simulator', type=str, nargs='?', default="SemiparametricQuadFormModel",
                        help="Simulator name")
    parser.add_argument('--model_type', type=str, nargs='?', default="full", help='Type of model')
    parser.add_argument('--model_update', type=int, nargs='?', default=10, help="Optimize model every k acquisitions")
    parser.add_argument('--model_iters', type=int, nargs='?', default=50, help="Number of iterations per update")
    parser.add_argument('--problem', choices=['pairs', 'quads', 'halve'], default='pairs', help="type of problem")
    parser.add_argument('--num_dims', type=int, nargs='?', default=8, help="dimension of the problem.")
    parser.add_argument('--path', type=str, nargs='?', default="storage/HBOLFI/", help="Results dir.")
    parser.add_argument('--acq_rule', type=str, nargs='?', default="LCB", help="Acquisition rule.")
    parser.add_argument('--acq_noise', type=float, nargs='?', default=1e-4,
                        help="Noise to be added in the acquisition step.")
    parser.add_argument('--seed', type=int, nargs='?', default=117, help="RNG seed")
    parser.add_argument('--corr', type=float, nargs='?', default=0.6, help="Correlation between variables in same block")
    parser.add_argument('--mu_min', type=float, nargs='?', default=0.5, help="Minimum of ground truth function")

    args = parser.parse_args()
    settings = vars(args)

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
    TRUE_KERNEL = settings['true_kernel']
    TRUE_PARTITION = settings['true_partition']
    SEARCH_SPACE = settings['search_space']

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
        true_model = QuadFormModel(kernel=TRUE_KERNEL, seed=SEED, **settings)



    true_loc = true_model.min_loc
    true_value = true_model.min_val
    assert(true_value == 0.)
    threshold = true_value  # + 2 * math.sqrt(settings['noise'])
    bolfi_gtruth = BOLFI(obj_function=true_model.observe, model=true_model,
                         search_space=SEARCH_SPACE, initial_samples=0)
    out = infer_distributions(bolfi=bolfi_gtruth, threshold=threshold, initial_params=true_loc,
                              num_groups=settings['num_groups'],
                              num_samples=settings['vi_num_samples'], maxiter=settings['vi_maxiter'],
                              with_vi=True, with_vi_no_prior=True, seed=SEED, vi_mode='oracle', verbose=True, name='ref')
    mvn_lap_ref, mvn_vi_ref, mvn_vi_llik_ref = out
    out = infer_distributions(bolfi=bolfi_gtruth, threshold=threshold, initial_params=true_loc,
                              num_groups=settings['num_groups'],
                              num_samples=settings['vi_num_samples'], maxiter=settings['vi_maxiter'],
                              with_vi=True, with_vi_no_prior=True, seed=SEED, vi_mode='full', verbose=True, name='ref')
    _, mvn_vi_ref_full, mvn_vi_llik_ref_full = out

    if DEBUG:
        orig = lambda x: bolfi_gtruth.unnormalized_loglikelihood(threshold, x.repeat(1, settings['num_groups']))
        lap = lambda x: mvn_lap_ref.log_prob(x.repeat(1, settings['num_groups']))
        vi = lambda x: mvn_vi_ref.log_prob(x.repeat(1, settings['num_groups']))
        vi_llik = lambda x: mvn_vi_llik_ref.log_prob(x.repeat(1, settings['num_groups']))

        from hbolfi.utils import plot_contour
        ax = plot_contour(orig, bounds=(-2.9, 2.9), colors='r')
        plot_contour(lap, bounds=(-2.9, 2.9), colors='g', ax=ax)
        plot_contour(vi, bounds=(-2.9, 2.9), colors='k', ax=ax)
        plot_contour(vi_llik, bounds=(-2.9, 2.9), colors='b', ax=ax)
        import matplotlib.pyplot as plt
        plt.show()
        print(mvn_lap_ref.covariance_matrix.diag())
        print("==========")
        print(mvn_vi_ref.covariance_matrix.diag())
        print("==========")
        print(mvn_vi_llik_ref.covariance_matrix.diag())

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
    print("Loaded test set")

    test_inputs = obj["samples"]
    test_mvn = bolfi_gtruth.model(test_inputs)
    test_targets = test_mvn.mean
    mll_ref = gpytorch.mlls.ExactMarginalLogLikelihood(bolfi_gtruth.model.likelihood, bolfi_gtruth.model.model)
    test_llik_ref = mll_ref(test_mvn, test_targets).item()

    idx_top = torch.argsort(test_targets)[:500]
    test_inputs_top = test_inputs[idx_top]
    test_targets_top = test_targets[idx_top]
    test_mvn_top = bolfi_gtruth.model(test_inputs_top)
    test_llik_ref_top = mll_ref(test_mvn_top, test_targets_top).item()

    print("Starting BOLFI")
    # Approximation
    mean_function = copy.deepcopy(true_model.mean_function)
    if SIMULATOR == 'SemiparametricQuadFormModel':
        init = {'lengthscale': settings['true_lengthscale'], 'outputscale': settings['true_outputscale'],
                'noise': settings['true_noise']}
    if MODEL_TYPE in ['mh', 'gibbs']:
        gpm = AdditiveStructureGPModel(kernel=INITIAL_KERNEL, mean=mean_function, optim_param=(), init=init)
    else:
        gpm = ExactGPModel(kernel=INITIAL_KERNEL, mean=mean_function, optim_param=(), init=init)
    bolfi = BOLFI(obj_function=true_model.observe, model=gpm, search_space=SEARCH_SPACE,
                  acquisition=ACQUISITION, seed=SEED)

    if MODEL_MODIFIER:
        selector = MODEL_MODIFIER(model=bolfi.model, seed=SEED)
        MODEL_MODIFIER = lambda: selector.run(MODEL_ITERS)
    for i in range(settings['num_acqs'] + 1):  # TODO: check posterior samples for dim==2 from pyro
        start_time = time.time()
        optimize = i % MODEL_UPDATE == 0
        bolfi.step(acquire=i > 0, optimize=optimize, model_modifier=MODEL_MODIFIER if optimize else None)
        # Compute stats
        if i % STATS_UPDATE == 0 and optimize:
            exp_loc_min, exp_val_min = bolfi.expected_min()
            loc_error = torch.max(torch.abs(exp_loc_min - true_loc)).item()
            regret = (true_model.observe_noiseless(exp_loc_min) - true_value).item()
            res_dict = OrderedDict(zip(("iter", "exp_loc_min", "exp_val_min", "loc_error", "regret"),
                                       (i, exp_loc_min.numpy(), exp_val_min, loc_error, regret)))
            res_dict = summary_stats(bolfi.model, test_inputs, test_targets, test_llik_ref, res_dict)
            res_dict = summary_stats(bolfi.model, test_inputs_top, test_targets_top, test_llik_ref_top,
                                     res_dict, suffix='_top')
            lb, ub = settings['search_space']['bounds']
            near_b = not (torch.all(lb + 0.1 <= exp_loc_min) and torch.all(exp_loc_min <= ub - 0.1))
            kl_lap, kl_vi, kl_vi_llik = (float('nan'), ) * 3
            kl_vi_full, kl_vi_llik_full = (float('nan'),) * 2
            if not near_b:
                out = infer_distributions(bolfi=bolfi, threshold=exp_val_min, initial_params=exp_loc_min,
                                          num_groups=settings['num_groups'],
                                          num_samples=settings['vi_num_samples'], maxiter=settings['vi_maxiter'],
                                          with_vi=True, with_vi_no_prior=True, seed=SEED, vi_mode='oracle', verbose=True,
                                          name='apprx')
                mvn_lap, mvn_vi, mvn_vi_llik = out
                kl_lap = kl_divergence(mvn_lap_ref, mvn_lap).item()
                kl_vi = kl_divergence(mvn_vi_ref, mvn_vi).item()
                kl_vi_llik = kl_divergence(mvn_vi_llik_ref, mvn_vi_llik).item()

                out = infer_distributions(bolfi=bolfi, threshold=exp_val_min, initial_params=exp_loc_min,
                                          num_groups=settings['num_groups'],
                                          num_samples=settings['vi_num_samples'], maxiter=settings['vi_maxiter'],
                                          with_vi=True, with_vi_no_prior=True, seed=SEED, vi_mode='full', verbose=True,
                                          name='apprx')
                _, mvn_vi_full, mvn_vi_llik_full = out
                kl_vi_full = kl_divergence(mvn_vi_ref_full, mvn_vi_full).item()
                kl_vi_llik_full = kl_divergence(mvn_vi_llik_ref_full, mvn_vi_llik_full).item()

            res_dict.update({'near_b': near_b, 'kl_lap': kl_lap, 'kl_vi': kl_vi, 'kl_vi_llik': kl_vi_llik})
            res_dict.update({'kl_vi_full': kl_vi_full, 'kl_vi_llik_full': kl_vi_llik_full})
            if isinstance(bolfi.model, AdditiveStructureGPModel):
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
