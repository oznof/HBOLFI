import os
import argparse
import copy
import csv
from collections import OrderedDict
from pathlib import Path
from time import time

import dill
import numpy as np

import torch
import gpytorch
from gpytorch.kernels import AdditiveKernel

from hbolfi import LCB, LCBAdd, EI, MHStructureSelector, GibbsStructureSelector
from hbolfi import ScaleRBFKernel, ScaleMaternKernel, CylindricalKernel
from hbolfi import settings as global_settings
from hbolfi import make_gp, BayesianOptimization, BayesianGPModel
from hbolfi import partition_metrics

from hbolfi.simulators import Branin, Hartmann6, Rosenbrock, Levy

# torch.autograd.set_detect_anomaly(True)
# cdist can cause sigfault
delattr(torch, 'cdist')

ACQUISITION_RULES = {"LCB": LCB, "LCBAdd": LCBAdd, "EI": EI}
MODEL_MODIFIERS = {"mh": MHStructureSelector, "gibbs": GibbsStructureSelector}
KERNEL_TYPES = {'ScaleRBFKernel': ScaleRBFKernel,
                'ScaleMaternKernel': ScaleMaternKernel,
                'CylindricalKernel': CylindricalKernel}
PROBLEMS = {'Branin': Branin, 'Hartmann6': Hartmann6, 'Rosenbrock': Rosenbrock, 'Levy': Levy}


def prepare_experiment(settings, make_dirs=True):
    torch.set_default_dtype(torch.float64)
    other_settings = dict(num_acqs=200, save_model_update=50,
                          min_bound=-1, max_bound=1)
    settings.update(other_settings)
    # Define search_space
    num_dims = settings['num_dims']
    lower_bound = torch.full((num_dims,), settings['min_bound'])
    upper_bound = torch.full((num_dims,), settings['max_bound'])
    bounds = torch.stack((lower_bound, upper_bound))
    settings['search_space'] = {'dim': num_dims, 'bounds': bounds}

    # Experiment description
    exp_keys = [settings[p] for p in ['num_dims', 'acq_rule', 'ard', 'n_init_eval', 'problem']]
    exp_name = "_".join(map(str, exp_keys))
    if settings['normalize']:
        exp_name += '_normalize'
    if settings['warping'] and settings['kernel_type'] == 'CylindricalKernel':
        exp_name += '_warping'
    exp_path = os.path.join(settings['path'], settings['problem'], exp_name)
    settings['exp_path'] = exp_path
    settings['exp_name'] = exp_name
    model_type = settings['model_type']

    global_settings.bo_normalize_targets._set_state(settings['normalize'])
    if not settings['warping']:
        global_settings.alpha.set('Delta', torch.tensor(1.))
        global_settings.beta.set('Delta', torch.tensor(1.))
    kernel_cls = KERNEL_TYPES[settings['kernel_type']]
    if model_type not in ['indep']:
        partition = [list(range(num_dims))]
        initial_kernel = kernel_cls(active_dims=torch.arange(num_dims),
                                    ard_num_dims=num_dims if settings['ard'] else None)
        if model_type in ['mh', 'gibbs']:
            initial_kernel = AdditiveKernel(initial_kernel)
    else:  # indep
        partition = [[v] for v in range(num_dims)]
        if not settings['ard']:
            initial_kernel = gpytorch.kernels.AdditiveStructureKernel(kernel_cls(), num_dims=num_dims)
        else:
            initial_kernel = AdditiveKernel(*[kernel_cls(active_dims=torch.tensor(dd)) for dd in range(num_dims)])

    if settings['kernel_type'] == 'CylindricalKernel':
        if model_type == 'full':
            from hbolfi import InvGammaPrior, invgamma_solver
            max_r = num_dims ** 0.5
            lp = InvGammaPrior(*invgamma_solver(max_r / 100., 2 * max_r, method='minimize'),
                               initial_value=torch.tensor(max_r))
            global_settings.lengthscale.set(prior=lp)
        else:
            raise NotImplementedError()

    initial_kernel.set_priors()
    settings['initial_partition'] = partition
    settings['initial_kernel'] = initial_kernel
    settings['acquisition'] = ACQUISITION_RULES[settings['acq_rule']]
    settings['model_modifier'] = MODEL_MODIFIERS.get(settings['model_type'])

    # Set experiment paths
    exp_path = settings['exp_path']
    model_name = f"{settings['model_type']}_{settings['kernel_type']}"
    if settings['model_type'] in ['mh', 'gibbs']:
        model_name += f"_{settings['model_update']}_{settings['model_iters']}"
    if settings['name_suffix']:
        model_name += f"_{settings['name_suffix']}"
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


if __name__ == '__main__':
    # ==================================================================================================================
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark BO")
    parser.add_argument('--model_type', type=str, nargs='?', default="full", help='Type of model')
    parser.add_argument('--kernel_type', type=str, nargs='?', default="RBFKernel", help='Type of kernel')
    parser.add_argument('--model_update', type=int, nargs='?', default=1, help="Update model every k acquisitions")
    parser.add_argument('--model_iters', type=int, nargs='?', default=50, help="Number of iterations per update")
    parser.add_argument('--problem', choices=['Branin', 'Hartmann6', 'Rosenbrock', 'Levy'],
                        default='Rosenbrock', help="type of problem")
    parser.add_argument('--num_dims', type=int, nargs='?', default=8, help="dimension of the problem.")
    parser.add_argument('--path', type=str, nargs='?', default="~/storage/HBOLFI/Baseline/", help="Results dir.")
    parser.add_argument('--acq_rule', type=str, nargs='?', default="LCB", help="Acquisition rule.")
    parser.add_argument('-n', '--n_init', dest='n_init_eval', type=int, default=20)
    parser.add_argument('--seed', type=int, nargs='?', default=117, help="RNG seed")
    parser.add_argument('--name_suffix', type=str, nargs='?', default='')
    parser.add_argument('--ard', dest='ard', action='store_true', default=False)
    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    parser.add_argument('--bayesian', dest='bayesian', action='store_true', default=False)
    parser.add_argument('-w', '--warping', dest='warping', action='store_true', default=False)
    parser.add_argument('--normalize', dest='normalize', action='store_true', default=False)


    args = parser.parse_args()
    settings = vars(args)
    DEBUG = settings['debug']
    settings['path'] = settings['path'].replace('~', str(Path.home()))
    settings = prepare_experiment(settings, make_dirs=not DEBUG)
    gpytorch.settings._fast_covar_root_decomposition._state = False
    gpytorch.settings._fast_log_prob._state = False
    gpytorch.settings._fast_solves._state = False
    gpytorch.settings.fast_pred_var._state = False
    gpytorch.settings.fast_pred_samples._state = False
    gpytorch.settings.lazily_evaluate_kernels._state = True
    gpytorch.settings.num_trace_samples._global_value = 0

    PROBLEM = settings['problem']
    EXP_PATH = settings['exp_path']
    EXP_NAME = settings['exp_name']
    STATS_PATH = settings['stats_path']
    SAVE_MODEL_PATH = settings['save_model_path']
    SAVE_MODEL_UPDATE = settings['save_model_update']
    SEED = settings["seed"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    NUM_DIMS = settings['num_dims']
    SEARCH_SPACE = settings['search_space']
    MODEL_TYPE = settings['model_type']
    MODEL_NAME = settings['model_name']
    MODEL_UPDATE = settings['model_update']
    MODEL_ITERS = settings['model_iters']
    MODEL_MODIFIER = settings['model_modifier']
    ACQUISITION = settings['acquisition']
    INITIAL_PARTITION = settings['initial_partition']
    INITIAL_KERNEL = settings['initial_kernel']
    print("Experiment: ", EXP_NAME)
    print("Save path: ", EXP_PATH)

    true_model = PROBLEMS[PROBLEM](NUM_DIMS)
    TRUE_PARTITION = true_model.partition
    true_loc = true_model.min_loc
    true_value = true_model.min_val
    if isinstance(INITIAL_KERNEL, CylindricalKernel):
        global_settings.quadpoly_a.set('Delta', torch.tensor(0.))
    global_settings.quadpoly_b.set('Delta', torch.tensor(0.))
    if settings['normalize']:
        global_settings.quadpoly_c.set('Delta', torch.tensor(0.))

    mean_function = None  # use QuadPolyMean
    if settings['bayesian']:
        # keep the parameter states that have been sampled in previous rounds
        global_settings.initialize_before_fitting._state = False
    gpm = make_gp(kernel=INITIAL_KERNEL, mean=mean_function, bayesian=settings['bayesian'])

    train_inputs = None
    #if settings['n_init_eval'] == 2 and isinstance(INITIAL_KERNEL, CylindricalKernel):
    #    train_inputs = torch.stack([torch.zeros(NUM_DIMS), torch.empty(NUM_DIMS).uniform_(-1, 1)])
    bo = BayesianOptimization(obj_function=true_model.eval, model=gpm, search_space=SEARCH_SPACE,
                              acquisition=ACQUISITION, seed=SEED, set_cube_transform=False,
                              initial_samples=settings['n_init_eval'], train_inputs=train_inputs)
    if MODEL_MODIFIER:
        selector = MODEL_MODIFIER(model=bo.model, seed=SEED)
        MODEL_MODIFIER = lambda model=None: selector.run(model, MODEL_ITERS)
    for i in range(settings['num_acqs'] + 1):
        start_time = time()
        fit = i % MODEL_UPDATE == 0
        #try:
        if isinstance(bo.model, BayesianGPModel) and bo.model.sampler == 'slice':
            if i == 0:
                bo.model.sampler_options['warmup_steps'] = 100
            else:
                bo.model.sampler_options['warmup_steps'] = 0
        bo.step(acquire=i > 0, fit=fit, model_modifier=MODEL_MODIFIER if fit else None,
                linear_update=False, verbose=True)
        # for i, sample in enumerate(bo.model.samples):
        #     print(i)
        #     print(sample.train_inputs[0])
        #     print(sample.train_targets)
        #     print(sample.param_state)
        #     print("===========")
        # import sys
        # sys.exit(1)
        #except RuntimeError:
        #    print(f'lengthscale={bo.model.kernel.radial_base_kernel.lengthscale.data}')
        #    print(f'outputscale={bo.model.kernel.radial_base_kernel.outputscale.data}')
        #    print(f'alpha={bo.model.kernel.alpha}, beta={bo.model.kernel.beta}')
        #    print(f'noise={bo.model.likelihood.noise.data}')
        end_time = time()
        print(f"target={bo.train_targets[-1]}, input={bo.train_inputs[-1]}, true_loc={true_loc}")
        if fit:
            if not isinstance(bo.model.kernel, CylindricalKernel):
                print(f'lengthscale={bo.model.kernel.lengthscale.data}')
                print(f'outputscale={bo.model.kernel.outputscale.data}')
            else:
                print(f'lengthscale={bo.model.kernel.radial_base_kernel.lengthscale.data}')
                print(f'outputscale={bo.model.kernel.radial_base_kernel.outputscale.data}')
                print(f'alpha={bo.model.kernel.alpha}, beta={bo.model.kernel.beta}')
            print(f'noise={bo.model.likelihood.noise.data}')
        #except:
        #    print(f'lengthscale={bo.model.kernel.lengthscale.data}')
        #    print(f'outputscale={bo.model.kernel.outputscale.data}')
        lb, ub = settings['search_space']['bounds']

        min_ind = torch.argmin(bo.train_targets)
        curr_min = bo.train_targets[min_ind]
        regret = (curr_min - true_value).item()
        if isinstance(curr_min, torch.Tensor):
            curr_min = curr_min.item()
        min_loc = bo.train_inputs[min_ind]
        min_loc_r = min_loc.norm().item()
        output_ = bo.train_targets[-1].item()
        input_ = bo.train_inputs[-1]
        input_r = input_.norm().item()
        loc_error = torch.max(torch.abs(min_loc - true_loc)).item()
        near_b = not (torch.all(lb + 0.1 <= min_loc) and torch.all(min_loc <= ub - 0.1))

        exp_loc_min, exp_val_min = None, None #bo.expected_min()
        exp_loc_min_r = None #exp_loc_min.norm().item()
        exp_regret = None #(true_model.eval(exp_loc_min) - true_value).item()
        exp_loc_error = None #torch.max(torch.abs(exp_loc_min - true_loc)).item()
        exp_near_b = None #not (torch.all(lb + 0.1 <= exp_loc_min) and torch.all(exp_loc_min <= ub - 0.1))

        res_dict = OrderedDict(iter=i, time=round(end_time - start_time, 3), curr_min=curr_min,
                               min_ind=min_ind.item(), min_loc_r=min_loc_r, output=output_, input_r=input_r,
                               neab_b=near_b, loc_error=loc_error, input=input_.numpy(), min_loc=min_loc,
                               exp_curr_min=exp_val_min, exp_regret=exp_regret, exp_loc_min_r=exp_loc_min_r,
                               exp_near_b=exp_near_b, exp_loc_error=exp_loc_error, exp_loc_min=exp_loc_min)
        print_dict = copy.deepcopy(res_dict)
        for key_to_remove in ['input', 'min_loc', 'exp_loc_min']:
            print_dict.pop(key_to_remove)
        print(print_dict)
        if isinstance(bo.model.kernel, AdditiveKernel):
            partition = bo.model.additive_structure
        else:
            partition = INITIAL_PARTITION
        if not isinstance(true_model, Rosenbrock):
            res_dict = partition_metrics(TRUE_PARTITION, partition, res_dict=res_dict)
        if not DEBUG:
            write_mode = 'a' if i > 0 else 'w'
            with open(os.path.join(STATS_PATH, f"{EXP_NAME}_{MODEL_NAME}_{SEED}.csv"), write_mode) as f:
                w = csv.DictWriter(f, res_dict.keys())
                if i == 0:
                    w.writeheader()
                w.writerow(res_dict)
            if i % SAVE_MODEL_UPDATE == 0:
                with open(os.path.join(SAVE_MODEL_PATH, f"{EXP_NAME}_{MODEL_NAME}_{i}.pkl"), "wb") as f:
                    dill.dump({"bo": bo}, f)
        print(f"Iteration {i:4}  Done. total time={time() - start_time:.3f}s")
