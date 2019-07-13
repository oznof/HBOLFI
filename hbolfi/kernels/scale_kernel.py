import math
from typing import Dict, Optional

import torch
from torch.nn import Module
import gpytorch
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel, RBFKernel, MaternKernel

from .. import settings


def make_scaled(cls: Kernel):
    def __init__(self,
                 priors: Optional[Dict[str, Prior]] = None,
                 set_priors: bool = True,
                 **kwargs):
        cls.__init__(self, **kwargs)
        outputscale = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))
        if priors is not None or set_priors:
            self.set_priors(priors)

    def outputscale_getter(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def outputscale_setter(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                last_dim_is_batch: bool = False, diag: bool = False, **params) -> torch.Tensor:
        orig_output = cls.forward(self, x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        outputscales = self.outputscale
        if last_dim_is_batch:
            outputscales = outputscales.unsqueeze(-1)
        if diag:
            outputscales = outputscales.unsqueeze(-1)
            return gpytorch.lazy.delazify(orig_output) * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            return orig_output.mul(outputscales)

    def set_priors(self, priors: Optional[Dict[str, Prior]] = None, with_default=True,
                   without_initial_value=False) -> Kernel:
        if priors is None:
            priors = {}
        if with_default or 'outputscale_prior' in priors:
            prior = priors.get('outputscale_prior', settings.outputscale.prior())
            self.register_prior('outputscale_prior', prior, lambda: self.outputscale,
                                lambda v: self._set_outputscale(v))
            if without_initial_value:
                init_value = prior.constraint.initial_value
                prior.constraint.initial_value = None
            self.register_constraint('raw_outputscale', prior.constraint)
            if without_initial_value:
                prior.constraint.initial_value = init_value
        cls.set_priors(self, priors=priors, with_default=with_default, without_initial_value=without_initial_value)
        return self

    def out_of_bounds(self):
        return cls.out_of_bounds(self) or (self.outputscale < math.exp(-12)).any() or \
               (self.outputscale > math.exp(20)).any()

    def extra_repr(self) -> str:
        return f"lengthscale: {self.lengthscale.detach().numpy()}\n" \
               f"outputscale: {self.outputscale.detach().numpy()}"

    methods_to_add = {'__init__': __init__, '_set_outputscale': outputscale_setter,
                      'forward': forward, 'set_priors': set_priors, 'out_of_bounds': out_of_bounds,
                      'extra_repr': extra_repr}
    cls_scaled = type(f"Scale{cls.__name__}", (cls, ), methods_to_add)
    cls_scaled.outputscale = property(outputscale_getter, outputscale_setter)
    return cls_scaled


ScaleRBFKernel = make_scaled(RBFKernel)
ScaleMaternKernel = make_scaled(MaternKernel)
