#!/usr/bin/env python3
import time
from typing import Any, Dict, Tuple, Optional
from collections import OrderedDict
import copy

import numpy as np
import scipy
import scipy.optimize

import torch
from torch.distributions import Transform
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import Prior
from pyro.distributions import Delta

from ..surrogate_model import SurrogateModel
from ... import settings
from ...means import QuadPolyMean
# TODO: REMOVE THIS
from ...kernels import ScaleRBFKernel, CylindricalKernel
from ...kernels.cylindrical import original_prediction_strategy_cylindrical, original_training_strategy_cylindrical


class ExactGPModel(ExactGP, SurrogateModel):
    def __init__(self, kernel: Kernel,
                 train_inputs: Optional[torch.Tensor] = None,
                 train_targets: Optional[torch.Tensor] = None,
                 likelihood: Optional[GaussianLikelihood] = None,
                 mean: Optional[Mean] = None,
                 priors: Optional[Dict[str, Prior]] = None,
                 optimizer: Optional[str] = None,
                 maxiter: Optional[int] = None,
                 optimizer_options: Optional[Dict[str, Any]] = None,
                 to_centered_cube_transform: Optional[Transform] = None):
        if priors is None:
            priors = {}
        if likelihood is None:
            noise_prior = priors.get('noise_prior', settings.noise.prior())
            likelihood = GaussianLikelihood(noise_prior=noise_prior, noise_constraint=noise_prior.constraint)
        super().__init__(train_inputs, train_targets, likelihood)
        self.priors = priors
        if mean is None:
            input_dim = kernel.active_dims.numel()
            mean = QuadPolyMean(input_dim, priors=priors)
        self.mean = mean
        self.kernel = kernel
        self.optimizer = settings.default_optimizer.value() if optimizer is None else optimizer
        maxiter = settings.default_maxiter.value() if maxiter is None else maxiter
        if not optimizer_options:
            if self.optimizer in settings.torch_optimizers:
                optimizer_options = settings.default_torch_optimizer_options[self.optimizer]
            elif self.optimizer == 'LBFGS':
                optimizer_options = {'maxiter': maxiter}
            else:
                raise RuntimeError("Invalid optimizer.")
        self.maxiter = optimizer_options.get('maxiter', maxiter)
        self.to_centered_cube_transform = to_centered_cube_transform
        self.optimizer_options = optimizer_options
        self.log_likelihood = None  # only updated, when fitted

    def __call__(self, *args, **kwargs):
        if isinstance(self.kernel, CylindricalKernel):
            if self.train_inputs is not None and self.train_targets is not None:
                if self.training:
                    inputs = args[0]
                    train_inputs = self.train_inputs[0]
                    mask_zero = torch.all(train_inputs == 0, dim=-1)
                    has_zero = torch.any(mask_zero)
                    if has_zero:
                        self.prediction_strategy = None
                        # not supporting batches yet
                        assert(train_inputs.ndim == 2)
                        assert(torch.equal(train_inputs, inputs))
                        if settings.cylindrical_original_training_strategy.off():
                            means = []
                            covariances = []
                            for i, x in enumerate(train_inputs[~mask_zero]):
                                x = x.view(1, -1)
                                mvn = super().__call__(*args, xtest=x, **kwargs)
                                means.append(mvn.mean)
                                covariances.append(mvn.covariance_matrix)
                            mean = torch.stack(means).mean(0)
                            covariance = torch.stack(covariances).mean(0)
                            return MultivariateNormal(mean, covariance)
                        else:
                            return original_training_strategy_cylindrical(self, train_inputs)
                elif gpytorch.settings.prior_mode.off():
                    if settings.cylindrical_original_prediction_strategy.off():
                        # posterior mode and CylindricalKernel
                        # cannot use caches (prediction_strategy)
                        # must set train_train_covar based on test_points
                        # NOTE: Only computes variances (not covariances) and assuming 2D inputs (slow version)
                        inputs = list(args[0].view(-1, self.train_inputs[0].size(-1)))
                        mean = torch.zeros(len(inputs))
                        variance = torch.zeros(len(inputs))
                        for i, xtest in enumerate(inputs):
                            self.prediction_strategy = None
                            xtest = xtest.view(1, -1)
                            mvn = super().__call__(xtest, xtest=xtest, **kwargs)
                            mean[i] = mvn.mean
                            variance[i] = mvn.variance
                        return MultivariateNormal(mean, torch.diag(variance))
                    else:
                        # use this method by default
                        # original prediction method provided by HyperSphere (CylindricalKernel)
                        # was extended to compute covariances
                        return original_prediction_strategy_cylindrical(self,
                                                                        args[0].view(-1, self.train_inputs[0].size(-1)))
        return super().__call__(*args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> MultivariateNormal:
        x_transf = x if self.to_centered_cube_transform is None else self.to_centered_cube_transform(x)
        mean_x = self.mean(x_transf)
        covar_x = self.kernel(x_transf, **kwargs)
        return MultivariateNormal(mean_x, covar_x)

    def load_init(self):
        if settings.initialize_before_optimization.on():
            if settings.initialize_by_sampling.on():
                # explicitly iterating over these submodules allows to define other submodules for which
                # we do not want to initialize, e.g. can create copies to save states
                for submodule in [self.likelihood, self.mean, self.kernel]:
                    for prior_name, _, _, _ in submodule.named_priors():
                        submodule.sample_from_prior(prior_name)
            else:
                for submodule in [self.likelihood, self.mean, self.kernel]:
                    for prior_name, prior, _, _ in submodule.named_priors():
                        submodule.initialize(**{prior_name.replace('_prior', '').replace('raw_', ''):
                                                prior.constraint.initial_value})

    def fit(self, train_inputs: Optional[torch.Tensor] = None,
            train_targets: Optional[torch.Tensor] = None, verbose: bool = False):
        if verbose:
            print("Optimizing hyperparameters...", end='')
            start_time = time.time()
        self.load_init()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        if train_inputs is not None and train_targets is not None:
            self.set_train_data(inputs=train_inputs, targets=train_targets, strict=False)
        self.train()
        if self.optimizer == 'LBFGS':
            self.optimize_lbfgs(verbose=verbose)
        else:
            params = []
            for name, prior, _, _ in self.named_priors():
                if not isinstance(prior, Delta):
                    name = name.replace('_prior', '')
                    (*submodules, name) = name.split('.')
                    module = self
                    for submodule in submodules:
                        module = getattr(module, submodule)
                    params.append(getattr(module, 'raw_' + name))
            if params:
                optimizer = settings.torch_optimizers[self.optimizer](params, **self.optimizer_options)

                def closure() -> float:
                    optimizer.zero_grad()
                    output = self(self.train_inputs[0])
                    loss = -mll(output, self.train_targets)
                    loss.backward()
                    return loss.item()

                prev_loss = 0
                for _ in range(self.maxiter):
                    curr_loss = optimizer.step(closure)
                    # if verbose:
                    #     print(f"iter {_}: loss={curr_loss}")
                    ftol = abs(prev_loss - curr_loss) / max(1, abs(prev_loss), abs(curr_loss)) if prev_loss else 1
                    if ftol <= 1e-9:
                        break
                    prev_loss = curr_loss
                    # gtol = torch.cat([v.grad.view(-1) for v in params]).abs().sum()
                    # if gtol <= 1e-5:
                    #    break
        self.log_likelihood = mll(self(self.train_inputs[0]), self.train_targets).item() * self.train_targets.size(-1)
        self.eval()
        if verbose:
            print(f"Done. time={time.time()-start_time:.3f}. LogLikelihood={self.log_likelihood}")
            # TODO: remove this
            if isinstance(self.kernel, CylindricalKernel):
                print(f"outputscale={self.kernel.radial_base_kernel.outputscale}, "
                      f"lengthscale={self.kernel.radial_base_kernel.lengthscale}")
            elif isinstance(self.kernel, ScaleRBFKernel):
                print(f"outputscale={self.kernel.outputscale}, "
                      f"lengthscale={self.kernel.lengthscale}")
            print(f"mean_a={self.mean.a}, mean_b={self.mean.b}, mean_c={self.mean.c}")
        return self

    def optimize_lbfgs(self, verbose: bool = False):
        # (raw) unconstrained parameter names
        param_state = OrderedDict()
        for name, prior, _, _ in self.named_priors():
            name = name.replace('_prior', '')
            (*submodules, name) = name.split('.')
            pname = 'raw_' + name
            module = self
            for submodule in submodules:
                module = getattr(module, submodule)
            param = getattr(module, pname)
            if isinstance(prior, Delta):
                param.requires_grad_(False)
            else:
                param_state[pname] = param.requires_grad_(True)
        if param_state:
            mll = ExactMarginalLogLikelihood(self.likelihood, self)

            def fun_and_grad(raw_x: np.ndarray) -> Tuple[float, np.ndarray]:
                self.zero_grad()
                self.load_array_state(raw_x, param_state)
                if verbose:
                    print(param_state)
                output = self(self.train_inputs[0])
                loss = -mll(output, self.train_targets)
                loss.backward()
                grad = self.state_to_array(param_state, grad=True)
                if verbose:
                    print(loss.item())
                    print([v.grad for v in param_state.values()])
                return loss.item(), grad

            raw_x0 = self.state_to_array(param_state)
            res = scipy.optimize.minimize(fun_and_grad, raw_x0, method='L-BFGS-B', jac=True,
                                          options=self.optimizer_options)
            self.load_array_state(res['x'], param_state)

    @staticmethod
    def state_to_array(state: OrderedDict, grad: bool = False):
        if not grad:
            return np.concatenate([v.detach().clone().view(-1).cpu().double().numpy() for v in state.values()])
        return np.concatenate([v.grad.detach().clone().view(-1).cpu().double().numpy() for v in state.values()])

    @staticmethod
    def load_array_state(array: np.ndarray, state: OrderedDict):
        with torch.no_grad():
            i = 0
            for p, v in state.items():
                v.copy_(torch.tensor(array[i:i+v.numel()]).view(v.shape))
                i += v.numel()

    def get_fantasy_model(self, inputs, targets, **kwargs):
        if isinstance(self.kernel, CylindricalKernel):
            # NOTE: do not call ExactGP.get_fantasy_model for CylindricalKernel
            # prediction_strategy will be reset
            # this is used to store L_not0_not0 if settings.cylindrical_original_pred_method.on(),
            # see kernels.CylindricalKernel
            model = self.clone()
            train_inputs = torch.cat([self.model.train_inputs[0], inputs])
            train_targets = torch.cat([self.model.train_targets, targets])
            model.set_train_data(inputs=train_inputs, targets=train_targets, strict=False)
        else:
            model = ExactGP.get_fantasy_model(self, inputs, targets, **kwargs)
            # fix some bad references that happen inside get_fantasy_model
            model.likelihood.set_priors(self.priors, without_initial_value=True)
            model.prediction_strategy.likelihood = model.likelihood
            model.kernel.set_priors(model.priors, without_initial_value=True)
            model.mean.set_priors(model.priors, without_initial_value=True)
        return model

    def clone(self):
        # some attributes are not deep copied and
        # priors are set with defaults (unless specified during __init__) because they contain
        # references to the original object through closures.
        # in general, it's difficult to know exactly what transformations were defined
        # use prior context managers or specify the priors dict to get exact clones
        train_inputs, train_targets = self.train_inputs, self.train_targets
        to_centered_cube_transform, likelihood = self.to_centered_cube_transform, self.likelihood
        self.train_inputs, self.train_targets = (None, ) * 2
        self.to_centered_cube_transform, self.likelihood = (None, ) * 2
        new = copy.deepcopy(self)
        self.train_inputs, self.train_targets = train_inputs, train_targets
        self.to_centered_cube_transform, self.likelihood = to_centered_cube_transform, likelihood
        new.train_inputs, new.train_targets = train_inputs, train_targets
        new.to_centered_cube_transform = to_centered_cube_transform
        new.likelihood = copy.deepcopy(self.likelihood).set_priors(new.priors, without_initial_value=True)
        new.kernel.set_priors(new.priors, without_initial_value=True)
        new.mean.set_priors(new.priors, without_initial_value=True)
        return new

    def __copy__(self):
        return self.clone()

    def _check_prior_setters(self):
        for prior_name, prior, closure, setting_closure in self.named_priors():
            name = prior_name.replace('_prior', '')
            constraint = self.constraint_for_parameter_name(name)
            if constraint is None:
                names = name.split('.')
                raw_name = '.'.join(names[:-1] + ['raw_' + names[-1]])
                constraint = self.constraint_for_parameter_name(raw_name)
            orig_val = closure().detach().clone()
            new_val = constraint.transform(constraint.inverse_transform(orig_val) + 1.17)
            setting_closure(new_val)
            attr = self
            for m in name.split('.'):
                attr = getattr(attr, m)
            if not torch.allclose(attr, new_val):
                raise RuntimeError(f"{prior_name} has not been set correctly.")
            setting_closure(orig_val)
        return True
