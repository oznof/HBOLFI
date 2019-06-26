#!/usr/bin/env python3
from typing import List, Dict, Optional

import torch
from gpytorch.kernels import AdditiveKernel
from gpytorch.priors import Prior
from gpytorch.models.exact_prediction_strategies import prediction_strategy

from .exact_gp import ExactGPModel
from ...means import QuadPolyMean
from ...kernels import ScaleRBFKernel


class AdditiveStructureGPModel(ExactGPModel):
    def __init__(self, kernel: AdditiveKernel, *args,  mean: Optional[QuadPolyMean] = None,
                 priors: Optional[Dict[str, Prior]] = None, **kwargs):
        if not isinstance(kernel, AdditiveKernel):
            raise RuntimeError("AdditiveStructureGPModel can only take in an AdditiveKernel")
        if mean is None:
            input_dim = torch.unique(torch.cat([k.active_dims for k in kernel.kernels])).numel()
            mean = QuadPolyMean(input_dim)
        super().__init__(kernel, *args, mean=mean, **kwargs)
        self.priors = priors if priors else {}
        self._orig_kernel = kernel

    def set_kernel(self, kernel: AdditiveKernel):
        self.kernel = kernel
        self._orig_kernel = kernel

    @property
    def additive_structure(self) -> List[torch.Tensor]:
        structure = [k.active_dims for k in self.kernel.kernels]
        return structure

    @property
    def is_restricted(self) -> bool:
        return self.kernel != self._orig_kernel

    def __call__(self, *args, **kwargs):
        # if model is restricted and in posterior mode, get unrestricted model to precompute train_train_covar
        if self.is_restricted and not self.training and \
           (self.train_inputs is not None and self.train_targets is not None):  # restricted and posterior mode
            train_inputs = self.train_inputs
            # if prediction_strategy has not yet been set, get unrestricted model to precompute train_train_covar
            if self.prediction_strategy is None:  # is set to None after each set_train_data
                restricted_kernel = self.kernel
                self.kernel = self._orig_kernel
                self.mean.active_dims = None
                train_prior_dist = super().__call__(*train_inputs, **kwargs)
                self.prediction_strategy = prediction_strategy(train_inputs=train_inputs,
                                                               train_prior_dist=train_prior_dist,
                                                               train_labels=self.train_targets,
                                                               likelihood=self.likelihood)
                self.kernel = restricted_kernel
                self.mean.active_dims = restricted_kernel.active_dims
        return super().__call__(*args, **kwargs)

    def load_init(self):
        self.kernel.set_priors(self.priors)
        super().load_init()

    def restrict(self, active_group: int):
        if not self.is_restricted:
            assert(isinstance(self.mean, QuadPolyMean))
            self.mean.num_groups = len(self._orig_kernel.kernels)
        self.kernel = self._orig_kernel.kernels[active_group]
        active_dims = self.kernel.active_dims
        self.mean.active_dims = active_dims

    def unrestrict(self):
        if self.is_restricted:
            self.kernel = self._orig_kernel
            self.mean.num_groups = None
            self.mean.active_dims = None


    #gg.eval()
    #print([p for p,v in gg.named_parameters()])
    #print([v for p, v in gg.named_parameters() if any([o in p for o in ['noise']])])
    #aa = QuadPolyMean(4, priors={'a_prior': , 'b_prior': 'c_prior': })
    #print(list(aa.named_priors()))
    #gg.restrict(0)

# THIS MUST GO TO MAIN
    # def set_init_and_priors(self, bounds):
    #     if self.ard:
    #         lengthscale = []
    #         lprior_params = []
    #         for l, u in bounds:
    #             lengthscale.append((u - l) / 3.)
    #             lprior_params.append(invgamma_solver((u - l) / 100., u - l))
    #             lengthscale = torch.Tensor(lengthscale)
    #     else:
    #         l = np.min(self.bounds)
    #         u = np.max(self.bounds)
    #         lengthscale = (u - l) / 3.
    #         lprior_params = [[invgamma_solver((u - l) / 100., u - l)]]
    #     if 'lengthscale' not in self.init:
    #         self.init['lengthscale'] = lengthscale
    #         self.optim_param.append('lengthscale')
    #     lprior_params = torch.Tensor(lprior_params)
    #     lprior = InvGammaPrior(
    #         alpha=lprior_params[:, 0], beta=lprior_params[:, 1])
    #     self.priors.setdefault('lengthscale', lprior)
    #     if 'outputscale' not in self.init:
    #         self.optim_param.append('outputscale')
    #     # mean 100, variance 100**2
    #     self.priors.setdefault('outputscale', GammaPrior(1, 0.01))
    #     self.priors.setdefault('noise', LogNormalPrior(0, 1))
