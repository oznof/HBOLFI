#!/usr/bin/env python3
from typing import Any, Dict, Tuple, Optional, Sequence
from collections import OrderedDict

import numpy as np
import scipy
import scipy.optimize

import torch
from torch.optim import Adam, SGD
from torch.distributions import Transform
from gpytorch.models import ExactGP
from gpytorch.means import Mean, ConstantMean
from gpytorch.kernels import Kernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import Prior

from ..surrogate_model import SurrogateModel
from ...distributions import DeltaPrior
from ...means import QuadPolyMean

from hbolfi import settings

OPTIMIZERS = {'Adam': Adam, 'SGD': SGD}  # only need to list pytorch optimizers, LBFGS is from scipy
DEFAULT_OPTIMIZER = 'LBFGS'
DEFAULT_OPTIMIZER_OPTIONS = {'Adam': {'lr': 0.01, 'amsgrad': False}, 'SGD': {'lr': 0.1, 'momentum': 0.9}}
DEFAULT_MAXITER = 1000


class ExactGPModel(ExactGP, SurrogateModel):
    def __init__(self, kernel: Kernel,
                 train_inputs: Optional[torch.Tensor] = None,
                 train_targets: Optional[torch.Tensor] = None,
                 likelihood: Optional[GaussianLikelihood] = None,
                 mean: Optional[Mean] = None,
                 priors: Optional[Dict[str, Prior]] = None,
                 optimizer: str = DEFAULT_OPTIMIZER,
                 maxiter: int = DEFAULT_MAXITER,
                 optimizer_options: Optional[Dict[str, Any]] = None,
                 to_centered_cube_transform: Optional[Transform] = None):
        if priors is None:
            priors = {}
        self.priors = priors
        if likelihood is None:
            noise_prior = priors.get('noise_prior', settings.noise.prior())
            likelihood = GaussianLikelihood(noise_prior=noise_prior, noise_constraint=noise_prior.constraint)
        super().__init__(train_inputs, train_targets, likelihood)
        if mean is None:
            input_dim = kernel.active_dims.numel()
            mean = QuadPolyMean(input_dim, priors=priors)
        self.mean = mean
        self.kernel = kernel
        self.optimizer = optimizer
        if not optimizer_options:
            if optimizer in OPTIMIZERS:  # pytorch optimizers
                optimizer_options = DEFAULT_OPTIMIZER_OPTIONS[optimizer]
            elif optimizer == 'LBFGS':  # uses scipy
                optimizer_options = {'maxiter': int(maxiter)}
            else:
                raise RuntimeError("Invalid optimizer.")
        self.maxiter = optimizer_options.get('maxiter', int(maxiter))
        self.to_centered_cube_transform = to_centered_cube_transform
        self.optimizer_options = optimizer_options
        self.log_likelihood = None  # only updated, when optimize is called

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        x_transf = x if self.to_centered_cube_transform is None else self.to_centered_cube_transform(x)
        mean_x = self.mean(x_transf)
        covar_x = self.kernel(x_transf)
        return MultivariateNormal(mean_x, covar_x)

    def load_init(self):
        if settings.initialize_before_optimization.on():
            for prior_name, _, _, _ in self.named_priors():
                self.sample_from_prior(prior_name)

    def optimize(self, train_inputs: Optional[torch.Tensor] = None,
                 train_targets: Optional[torch.Tensor] = None):
        self.load_init()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        if train_inputs is not None and train_targets is not None:
            self.set_train_data(inputs=train_inputs, targets=train_targets, strict=False)
        self.train()
        if self.optimizer == 'LBFGS':
            self.optimize_lbfgs()
        else:
            params = []
            for name, prior, _, _ in self.named_priors():
                if not isinstance(prior, DeltaPrior):
                    name = name.replace('_prior', '')
                    (*submodules, name) = name.split('.')
                    module = self
                    for submodule in submodules:
                        module = getattr(module, submodule)
                    params.append(getattr(module, 'raw_' + name))
            if params:
                optimizer = OPTIMIZERS[self.optimizer](params, self.optimizer_options)

                def closure() -> float:
                    optimizer.zero_grad()
                    output = self(self.train_inputs[0])
                    loss = -mll(output, self.train_targets)
                    loss.backward()
                    return loss.item()

                prev_loss = None
                for _ in range(self.iters):
                    curr_loss = optimizer.step(closure).item()
                    ftol = abs(prev_loss - curr_loss) / max(1, abs(prev_loss), abs(curr_loss)) if prev_loss else 1
                    if ftol <= 1e-9:
                        break
                    prev_loss = curr_loss
                    # gtol = torch.cat([v.grad.view(-1) for v in params]).abs().sum()
                    # if gtol <= 1e-5:
                    #    break
        self.log_likelihood = mll(self(self.train_inputs[0]), self.train_targets).item() * self.train_targets.size(-1)
        self.eval()
        return self

    def optimize_lbfgs(self):
        # (raw) unconstrained parameter names
        param_state = OrderedDict()
        for name, prior, _, _ in self.named_priors():
            if not isinstance(prior, DeltaPrior):
                name = name.replace('_prior', '')
                (*submodules, name) = name.split('.')
                pname = 'raw_' + name
                module = self
                for submodule in submodules:
                    module = getattr(module, submodule)
                param_state[pname] = getattr(module, pname)
        if param_state:
            mll = ExactMarginalLogLikelihood(self.likelihood, self)

            def fun_and_grad(raw_x: np.ndarray) -> Tuple[float, np.ndarray]:
                self.zero_grad()
                self.load_array_state(raw_x, param_state)
                output = self(self.train_inputs[0])
                loss = -mll(output, self.train_targets)
                loss.backward()
                grad = self.state_to_array(param_state, grad=True)
                return loss.item(), grad

            raw_x0 = self.state_to_array(param_state)
            res = scipy.optimize.minimize(fun_and_grad, raw_x0, method='L-BFGS-B', jac=True,
                                          options=self.optimizer_options)
            self.load_array_state(res['x'], param_state)

    @staticmethod
    def state_to_array(state: OrderedDict, grad: bool = False):
        if not grad:
            return np.concatenate([v.detach().view(-1).cpu().double().clone().numpy() for v in state.values()])
        return np.concatenate([v.grad.detach().view(-1).cpu().double().clone().numpy() for v in state.values()])

    @staticmethod
    def load_array_state(array: np.ndarray, state: OrderedDict):
        i = 0
        for p, v in state.items():
            v.copy_(torch.tensor(array[i:v.numel()]))
            i += v.numel()
