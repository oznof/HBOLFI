import os
import copy
from time import time

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import AdditiveKernel

from hbolfi import ScaleRBFKernel, ScaleMaternKernel, CylindricalKernel
from hbolfi import LCB, LCBAdd
from hbolfi import MHStructureSelector, GibbsStructureSelector
from hbolfi import settings as global_settings


ACQUISITION_RULES = {"LCB": LCB, "LCBAdd": LCBAdd}
MODEL_MODIFIERS = {"mh": MHStructureSelector, "gibbs": GibbsStructureSelector}
KERNEL_TYPES = {'ScaleRBFKernel': ScaleRBFKernel,
                'ScaleMaternKernel': ScaleMaternKernel,
                'CylindricalKernel': CylindricalKernel}
DEFAULT_KERNEL_TYPE = 'ScaleRBFKernel'


def prepare_experiment(settings, make_dirs=True, scale_noise=False):
    torch.set_default_dtype(torch.float64)
    other_settings = dict(ard=False, num_acqs=500, stats_update=10, save_model_update=50,
                          noise=1e-2, min_bound=-1, max_bound=1)
    settings.update(other_settings)
    vi_settings = dict(vi_num_samples=5000, vi_maxiter=1000)
    nuts_settings = {'nuts_options': dict(num_samples=5000, num_chains=4, warmup_steps=250, num_thinning=2)}
    settings.update(vi_settings)
    settings.update(nuts_settings)

    num_dims = settings['num_dims']
    group_dims = {'pairs': 2, 'quads': 4, 'halve': settings['num_dims'] // 2}
    group_dims = group_dims[settings['problem']]
    settings['group_dims'] = group_dims
    num_groups = num_dims // group_dims
    settings['num_groups'] = num_groups

    if scale_noise:
        settings['noise'] *= num_groups ** 2

    # Define search_space
    lower_bound = torch.full((num_dims,), settings['min_bound'])
    upper_bound = torch.full((num_dims,), settings['max_bound'])
    bounds = torch.stack((lower_bound, upper_bound))
    settings['search_space'] = {'dim': num_dims, 'bounds': bounds}

    # Define ground truth partition and model parameters (e.g. kernel)
    true_partition = np.arange(num_groups * group_dims).reshape((num_groups, group_dims))
    true_partition = [list(pi) for pi in true_partition]
    settings['true_partition'] = true_partition
    if settings['simulator'] == 'SemiparametricQuadFormModel':
        # In this simulator, we do not optimize the hyperparameters, we set the same hypers as those in the
        # ground truth model, all priors are instances of DeltaPrior
        global_settings.fix_all_hyperparameters._state = True
        global_settings.noise.set('Delta', initial_value=torch.as_tensor(settings['noise']))
        settings.setdefault('true_kernel_type', DEFAULT_KERNEL_TYPE)
        true_kernel_cls = KERNEL_TYPES[settings['true_kernel_type']]
        true_kernel = [true_kernel_cls(active_dims=pi, ard_num_dims=len(pi) if settings['ard'] else None) for pi in
                       true_partition]
        true_kernel = AdditiveKernel(*true_kernel)
        true_kernel.set_priors()
        settings['true_kernel'] = true_kernel

    # Experiment description
    exp_keys = [settings[p] for p in ['num_dims', 'acq_rule', 'acq_noise', 'ard', 'problem']]
    exp_name = "_".join(map(str, exp_keys))
    exp_path = os.path.join(settings['path'], settings['simulator'], exp_name)
    settings['exp_path'] = exp_path
    settings['exp_name'] = exp_name
    model_type = settings['model_type']

    if model_type == 'oracle':
        if settings['simulator'] == 'SemiparametricQuadFormModel':
            settings['kernel_type'] = settings['true_kernel_type']
            partition = true_partition
            initial_kernel = copy.deepcopy(settings['true_kernel'])
        else:
            raise NotImplementedError()
    else:
        settings.setdefault('kernel_type', DEFAULT_KERNEL_TYPE)
        kernel_cls = KERNEL_TYPES[settings['kernel_type']]
        if model_type not in ['indep', 'indepscaled']:
            partition = [list(range(num_dims))]
            initial_kernel = kernel_cls(active_dims=range(num_dims), ard_num_dims=num_dims if settings['ard'] else None)
            if model_type == 'fullscaled':  # this only makes sense in the SemiparametricQuadFormModel
                outputscale_value = global_settings.outputscale.initial_value
                global_settings.outputscale.set(prior='Delta', initial_value=outputscale_value * num_groups)
            if model_type in ['mh', 'gibbs']:
                initial_kernel = gpytorch.kernels.AdditiveKernel(initial_kernel)
        else:  # indep and indepscaled
            partition = [[v] for v in range(num_dims)]
            if model_type == 'indepscaled':  # this only makes sense in the SemiparametricQuadFormModel
                outputscale_value = global_settings.outputscale.initial_value
                global_settings.outputscale.set(prior='Delta', initial_value=outputscale_value / group_dims)
            if not settings['ard']:
                initial_kernel = gpytorch.kernels.AdditiveStructureKernel(kernel_cls(), num_dims=num_dims)
            else:  # indep and indepscaled (non-shared parameters)
                initial_kernel = gpytorch.kernels.AdditiveKernel(*[kernel_cls(active_dims=[dd])
                                                                   for dd in range(num_dims)])
    initial_kernel.set_priors()
    settings['initial_partition'] = partition
    settings['initial_kernel'] = initial_kernel

    settings['acquisition'] = ACQUISITION_RULES[settings['acq_rule']]
    settings['model_modifier'] = MODEL_MODIFIERS.get(settings['model_type'])
    # Set experiment paths
    exp_path = settings['exp_path']
    model_name = settings['model_type']
    if model_name in ['mh', 'gibbs']:
        model_name += f"_{settings['model_update']}_{settings['model_iters']}"
    settings['model_name'] = model_name
    save_model_path = os.path.join(exp_path, model_name, str(settings['seed']))
    settings['save_model_path'] = save_model_path
    stats_path = os.path.join(exp_path, "stats")
    settings['stats_path'] = stats_path
    if make_dirs:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)
    return settings


def infer_distributions(bolfi, threshold, initial_params=None, num_groups=None, num_samples=5000, maxiter=1000,
                        with_vi=True, with_vi_no_prior=True, vi_mode='oracle', seed=None, verbose=False, name=''):
    out = []
    if verbose:
        print(f"Computing {name} posterior distribution with Laplace...", end='')
    start_time = time()
    mvn_lap = bolfi.infer(threshold, method='laplace', initial_params=initial_params, n_inits=1)
    out.append(mvn_lap)
    if verbose:
        print(f"Done.  time={time() - start_time:.3f}s.")
    if num_groups is None and (with_vi or with_vi_no_prior) and vi_mode == 'oracle':
        raise RuntimeError("Must provide num_groups to run gaussian_vi_oracle.")
    if with_vi:
        if verbose:
            print(f"Computing {name} posterior distribution with {vi_mode}GaussianVI...", end='')
        start_time = time()
        mvn_vi = bolfi.infer(threshold, method='gaussian_vi_' + vi_mode, initial_params=mvn_lap.mean,
                             num_groups=num_groups, n_inits=1,
                             num_samples=num_samples, maxiter=maxiter, seed=seed)
        out.append(mvn_vi)
        if verbose:
            print(f"Done.  time={time() - start_time:.3f}s.")

    if with_vi_no_prior:
        if verbose:
            print(f"Computing {name} distribution with {vi_mode}GaussianVI..", end='')
        start_time = time()
        mvn_vi_llik = bolfi.infer(threshold, method='gaussian_vi_' + vi_mode,
                                  initial_params=mvn_lap.mean, with_prior=False,
                                  num_groups=num_groups, n_inits=1,
                                  num_samples=num_samples, maxiter=maxiter, seed=seed)
        out.append(mvn_vi_llik)
        if verbose:
            print(f"Done.  time={time() - start_time:.3f}s.")
    if len(out) == 1:
        out = out[0]
    return out
