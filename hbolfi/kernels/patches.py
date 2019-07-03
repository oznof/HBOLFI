from typing import Dict, Optional

import torch
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel, AdditiveKernel, AdditiveStructureKernel

from .. import settings


def kernel_set_priors(self, priors: Optional[Dict[str, Prior]] = None, with_default=True,
                      without_initial_value=False) -> Kernel:
    if priors is None:
        priors = {}
    if self.has_lengthscale and (with_default or 'lengthscale_prior' in priors):
        prior = priors.get('lengthscale_prior',   settings.lengthscale.prior())
        self.register_prior('lengthscale_prior', prior, lambda: self.lengthscale, lambda v: self._set_lengthscale(v))
        if without_initial_value:
            init_value = prior.constraint.initial_value
            prior.constraint.initial_value = None
        self.register_constraint('raw_lengthscale', prior.constraint)
        if without_initial_value:
            prior.constraint.initial_value = init_value
    return self


# gpytorch.kernels.AdditiveKernel does not initialize properly
# the initialize method will be replaced by this function at run time
def additive_kernel_initialize(self, **kwargs):
    for name, val in kwargs.items():
        idx, name = name.split('.', 2)[1:]
        module = self.kernels[int(idx)]
        module.initialize(**{name: val})


def additive_kernel_set_priors(self, priors: Optional[Dict[str, Prior]] = None,
                               with_default=True, without_initial_value=False) -> AdditiveKernel:
    if priors is None:
        priors = {}
    for kernel in self.kernels:
        kernel.set_priors(priors=priors, with_default=with_default, without_initial_value=without_initial_value)
    return self


def additive_active_dims(self):
    return torch.unique(torch.cat([k.active_dims.view(-1) for k in self.kernels]))


def fully_additive_set_priors(self, priors: Optional[Dict[str, Prior]] = None,
                              with_default=True, without_initial_value=False) -> AdditiveStructureKernel:
    if priors is None:
        priors = {}
    self.base_kernel.set_priors(priors=priors, with_default=with_default,
                                without_initial_value=without_initial_value)
    return self
