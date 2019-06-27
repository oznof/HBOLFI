#!/usr/bin/env python3
from typing import List

import torch
from gpytorch.kernels import AdditiveKernel
from gpytorch.models.exact_prediction_strategies import prediction_strategy

from .exact_gp import ExactGPModel
from ...means import QuadPolyMean


class AdditiveStructureGPModel(ExactGPModel):
    def __init__(self, kernel: AdditiveKernel, *args, **kwargs):
        if not isinstance(kernel, AdditiveKernel):
            raise RuntimeError("AdditiveStructureGPModel can only take in an AdditiveKernel")
        super().__init__(kernel, *args, **kwargs)
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
