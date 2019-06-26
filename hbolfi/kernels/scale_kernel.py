from typing import Dict, Optional

import torch
import gpytorch
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel, RBFKernel, MaternKernel
from gpytorch.constraints import Positive

from hbolfi import settings


def make_scaled(cls: Kernel):
    def __init__(self,
                 outputscale_prior: Optional[Prior] = None,
                 outputscale_constraint: Optional[torch.nn.Module] = None,
                 **kwargs):
        cls.__init__(self, **kwargs)
        if outputscale_prior is not None:
            outputscale_constraint = outputscale_prior.constraint
        if outputscale_constraint is None:
            outputscale_constraint = Positive()
        outputscale = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))
        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior", outputscale_prior, lambda: self.outputscale, lambda v: self._set_outputscale(v)
            )
        self.register_constraint("raw_outputscale", outputscale_constraint)

    def outputscale_getter(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def outputscale_setter(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                last_dim_is_batch: bool = False, diag: bool = False, **params) -> torch.Tensor:
        orig_output = super().forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        outputscales = self.outputscale
        if last_dim_is_batch:
            outputscales = outputscales.unsqueeze(-1)
        if diag:
            outputscales = outputscales.unsqueeze(-1)
            return gpytorch.lazy.delazify(orig_output) * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            return orig_output.mul(outputscales)

    def set_priors(self, priors: Optional[Dict[str, Prior]] = None, with_default=True):
        if priors is None:
            priors = {}
        if with_default or 'outputscale_prior' in priors:
            prior = priors.get('outputscale_prior', settings.outputscale.prior())
            self.register_prior('outputscale_prior', prior, lambda: self.outputscale,
                                lambda v: self._set_outputscale(v))

            self.register_constraint('raw_outputscale', prior.constraint)
        cls.set_priors(self, priors=priors, with_default=with_default)

    def extra_repr(self) -> str:
        return f"lengthscale: {self.lengthscale.detach().numpy()}\n" \
               f"outputscale: {self.outputscale.detach().numpy()}"

    methods_to_add = {'__init__': __init__, '_set_outputscale': outputscale_setter,
                      'forward': forward, 'set_priors': set_priors, 'extra_repr': extra_repr}
    cls_scaled = type(f"Scale{cls.__name__}", (cls, ), methods_to_add)
    cls_scaled.outputscale = property(outputscale_getter, outputscale_setter)
    return cls_scaled


ScaleRBFKernel = make_scaled(RBFKernel)
ScaleMaternKernel = make_scaled(MaternKernel)