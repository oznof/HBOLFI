import unittest
import time

import torch
import gpytorch
from gpytorch.kernels import AdditiveKernel

from hbolfi.simulators import Branin, Hartmann6, Rosenbrock, Levy
from hbolfi import BayesianOptimization, LCB
from hbolfi import ScaleRBFKernel, ScaleMaternKernel, CylindricalKernel
from hbolfi import make_gp
from hbolfi import settings, rng

seed = 117
test_function = {'Branin': Branin, 'Hartmann6': Hartmann6, 'Rosenbrock': Rosenbrock, 'Levy': Levy}


def allclose_params(module, other, verbose=False):
    model_eq = module.__class__ == other.__class__
    if not model_eq:
        return False
    ret_val = True
    for (n1, p1), (n2, p2) in zip(module.named_parameters(), other.named_parameters()):
        if verbose:
            print(f"{n1}: {p1}")
            print(f"{n2}: {p2}")
            print("============")
        if not torch.allclose(p1, p2):
            ret_val = False
    return ret_val


class TestReproducibility(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        if hasattr(torch, 'cdist'):
            delattr(torch, 'cdist')
        gpytorch.settings._fast_covar_root_decomposition._state = False
        gpytorch.settings._fast_log_prob._state = False
        gpytorch.settings._fast_solves._state = False
        gpytorch.settings.fast_pred_var._state = False
        gpytorch.settings.fast_pred_samples._state = False
        gpytorch.settings.lazily_evaluate_kernels._state = True
        gpytorch.settings.num_trace_samples._global_value = 0

    def test_acquisition_bo(self):
        dim = 8
        num_acqs = 3
        test_class = Levy
        acquisition_class = LCB
        kernel_class = ScaleRBFKernel
        true_model = test_class(dim)
        bounds = torch.ones(2, dim)
        bounds[0] = -1
        search_space = {'dim': dim, 'bounds': bounds}
        states = []
        for i in range(2):
            kernel = kernel_class(active_dims=torch.arange(dim))
            if kernel_class == CylindricalKernel:
                settings.quadpoly_a.set('Delta', torch.tensor(0.))
            else:
                settings.quadpoly_a.set('HalfNormal', torch.tensor(1.))
            settings.quadpoly_b.set('Delta', torch.tensor(0.))
            mean_function = None  # use QuadPolyMean
            gpm = make_gp(kernel=kernel, mean=mean_function)
            bo = BayesianOptimization(obj_function=true_model.eval, model=gpm, search_space=search_space,
                                      acquisition=acquisition_class, seed=seed, set_cube_transform=True)
            for j in range(num_acqs):
                bo.step(acquire=True, fit=False, linear_update=False)
                states.append(bo.model.train_inputs[0])
        states_r1 = states[:len(states) // 2]
        states_r2 = states[len(states) // 2:]
        self.assertTrue(len(states_r1) == len(states_r2))
        for r1, r2 in zip(states_r1, states_r2):
            self.assertTrue(torch.equal(r1, r2))

    def test_load_init(self):
        dim = 8
        for kernel_class in [ScaleRBFKernel, ScaleMaternKernel, CylindricalKernel, AdditiveKernel]:
            for linear_update in [True, False]:
                for init_by_sampling in [True, False]:
                    settings.initialize_by_sampling._set_state(init_by_sampling)
                    states = []
                    for i in range(2):
                        if kernel_class == AdditiveKernel:
                            kernel = ScaleRBFKernel(active_dims=torch.arange(dim // 2)) + \
                                     ScaleRBFKernel(active_dims=torch.arange(dim // 2) + (dim // 2))
                        else:
                            kernel = kernel_class(active_dims=torch.arange(dim))
                        mean_function = None  # use QuadPolyMean
                        gpm = make_gp(kernel=kernel, mean=mean_function)
                        if linear_update:
                            gpm.set_train_data(inputs=torch.ones(1, dim) * 0.1,
                                               targets=torch.ones(1) * 10, strict=False)
                            gpm.eval()
                            gpm(torch.ones(1, dim) * 1e-2)
                            gpm_before = gpm
                            gpm = gpm.get_fantasy_model(inputs=torch.ones(1, dim) * 1e-3, targets=torch.ones(1))
                        else:
                            gpm_before = gpm.clone()
                            self.assertTrue(gpm_before._check_prior_setters())
                        self.assertTrue(gpm._check_prior_setters())
                        with rng(torch.Generator().manual_seed(seed)):
                            gpm.load_init()
                        if not init_by_sampling:
                            for name, p, cons in gpm.named_parameters_and_constraints():
                                param = cons.transform(p.detach())
                                init = torch.as_tensor(cons.initial_value).to(param).expand(param.shape)
                                self.assertTrue(torch.allclose(param, init))
                        else:
                            self.assertFalse(allclose_params(gpm_before, gpm))
                        states.append(gpm)
                    self.assertTrue(allclose_params(states[0], states[1]))

    def test_fit_bo(self):
        # this test shows that the fitted parameters and evidence set is practically equal whether
        # using linear updates or not. Results would be exactly the same (with no precision error)
        # if running again using the same mode, e.g. [True, True] or [False, False]
        dim = 8
        num_acqs = 3
        test_class = Levy
        acquisition_class = LCB
        verbose = False
        for kernel_class in [ScaleRBFKernel, ScaleMaternKernel, AdditiveKernel]:
            with self.subTest(kernel_class=kernel_class):
                true_model = test_class(dim)
                bounds = torch.ones(2, dim)
                bounds[0] = -1
                search_space = {'dim': dim, 'bounds': bounds}
                use_constrained = True
                acquisition_options = {'use_constrained': use_constrained, 'n_inits': 1}
                states = []
                if kernel_class == CylindricalKernel:
                    settings.quadpoly_a.set('Delta', torch.tensor(0.))
                    settings.alpha.set('Delta', torch.tensor(1.))
                    settings.beta.set('Delta', torch.tensor(1.))
                else:
                    settings.quadpoly_a.set('HalfNormal', torch.tensor(1.))
                settings.quadpoly_b.set('Delta', torch.tensor(0.))
                for flag in [False, True]:
                    if kernel_class == AdditiveKernel:
                        kernel = ScaleRBFKernel(active_dims=torch.arange(dim // 2)) + \
                                 ScaleRBFKernel(active_dims=torch.arange(dim // 2) + (dim // 2))
                    else:
                        kernel = kernel_class(active_dims=torch.arange(dim))
                    mean_function = None  # use QuadPolyMean
                    gpm = make_gp(kernel=kernel, mean=mean_function)
                    self.assertTrue(gpm.training)
                    bo = BayesianOptimization(obj_function=true_model.eval, model=gpm, search_space=search_space,
                                              acquisition=acquisition_class, acquisition_options=acquisition_options,
                                              seed=seed, set_cube_transform=True)
                    self.assertTrue(not gpm.training)
                    st = time.time()
                    for j in range(num_acqs):
                        if flag:
                            # if linear_update, must call model on data first (prediction)
                            # doing so initializes the prediction_strategy
                            with rng(torch.Generator().manual_seed(seed)):
                                bo.model(torch.randn(2, dim) / 1000.)
                        bo.step(acquire=True, fit=True, linear_update=flag)
                        if verbose:
                            print(kernel_class.__name__, bo.model.train_inputs[0][-1])
                        self.assertTrue(bo.model._check_prior_setters())
                        self.assertTrue(not bo.model.training)
                        states.append(bo.model.clone())
                    if verbose:
                        print(f"time linear_update={flag}: {time.time() - st:.3f}")
                states_r1 = states[:len(states) // 2]
                states_r2 = states[len(states) // 2:]
                self.assertTrue(len(states_r1) == len(states_r2))
                for i, (r1, r2) in enumerate(zip(states_r1, states_r2)):
                    with self.subTest(i=i):
                        self.assertTrue(allclose_params(r1, r2))
                        self.assertTrue(torch.allclose(r1.train_inputs[0], r2.train_inputs[0]))
                        self.assertTrue(torch.allclose(r1.train_targets, r2.train_targets))


if __name__ == '__main__':
    unittest.main()