from typing import Dict, Optional
import math

import torch
import torch.nn

from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from gpytorch import settings as gpysettings

from .. import settings
from ..utils import make_property
from ..kernels import ScaleMaternKernel
from ..distributions import DirichletPrior


# based to some extent on the implementation in gpytorch.kernels.CylindricalKernel
# IMPORTANT NOTE: we assume there is only one point at the origin (no repeats)
class CylindricalKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Cylindrical Kernel between
    inputs :math:`mathbf{x_1}` and :math:`mathbf{x_2}`.
    It was proposed in `BOCK: Bayesian Optimization with Cylindrical Kernels`.
    See http://proceedings.mlr.press/v80/oh18a.html for more details
    Args:
        :attr:`radial_base_kernel` (gpytorch.kernels.Kernel, optional):
            The base kernel for computing the radial kernel.
            Default: `MaternKernel(nu=5/2, lengthscale_constraint=Interval(max_radius / 100, 2 * max_radius))`
        :attr:`num_angular_weights` (int, optional):
            The number of components in the angular kernel. Default: `4`
        :attr:`max_radius` (float, optional):
            Radius of the n-ball. Default: `1.0`
        :attr:`batch_size` (int, optional):
            Set this if the data is batch of input data.
            It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `1`
        :attr:`eps` (float, optional):
            Small floating point number used to improve numerical stability
            in kernel computations. Default: `1e-6`
    """
    def __init__(self, radial_base_kernel: Optional[Kernel] = None, num_angular_weights: int = 4,
                 max_radius: float = 1.0, eps: float = 1e-6,
                 priors: Optional[Dict[str, Prior]] = None, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        if 'active_dims' in kwargs:
            max_radius = float(len(kwargs['active_dims'])) ** 0.5
        self.max_radius = max_radius
        self.num_angular_weights = num_angular_weights
        self.eps = eps
        self.radial_base_kernel = radial_base_kernel if radial_base_kernel else ScaleMaternKernel(nu=2.5)
        if priors is None:
            priors = {}
        angular_weights_prior = priors.get('angular_weights_prior', settings.angular_weights.prior())
        if isinstance(angular_weights_prior, DirichletPrior):
            num_angular_weights -= 1  # because it must be in simplex
        self.register_parameter(name="raw_angular_weights",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, num_angular_weights)))
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.set_priors(priors)

    # add attributes: constrained parameters
    angular_weights = make_property('raw_angular_weights')
    alpha = make_property('raw_alpha')
    beta = make_property('raw_beta')

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        r1 = torch.zeros(*x1.shape[:-1], 1)
        r2 = torch.zeros(*x2.shape[:-1], 1)
        mask_zero_1 = torch.all(x1 == 0, dim=-1)
        mask_zero_2 = torch.all(x2 == 0, dim=-1)
        mask_not_zero_1 = ~mask_zero_1
        mask_not_zero_2 = ~mask_zero_2
        r1[mask_not_zero_1] = x1[mask_not_zero_1, :].norm(dim=-1, keepdim=True)
        r2[mask_not_zero_2] = x2[mask_not_zero_2, :].norm(dim=-1, keepdim=True)
        r1[mask_zero_1] = x1[mask_zero_1, :].sum(dim=-1, keepdim=True) + self.eps
        r2[mask_zero_2] = x2[mask_zero_2, :].sum(dim=-1, keepdim=True) + self.eps

        if torch.any(r1 > self.max_radius) or torch.any(r2 > self.max_radius):
            raise RuntimeError(f"Cylindrical kernel not defined for data points with radius > {self.max_radius}")

        a1 = torch.zeros_like(x1)
        a2 = torch.zeros_like(x2)
        a1[mask_not_zero_1, :] = x1[mask_not_zero_1, :] / r1[mask_not_zero_1]
        a2[mask_not_zero_2, :] = x2[mask_not_zero_2, :] / r2[mask_not_zero_2]
        a1[mask_zero_1] = x1[mask_zero_1, :] + 1 / math.sqrt(x1.size(-1))
        a2[mask_zero_2] = x2[mask_zero_2, :] + 1 / math.sqrt(x2.size(-1))

        if not diag:
            gram_mat = a1.matmul(a2.transpose(-2, -1))
            gram_mat[mask_zero_1] = a2[torch.any(mask_zero_1, dim=-1)].mul(a2[torch.any(mask_zero_1, dim=-1)]).sum(-1)
            gram_mat[mask_zero_2.unsqueeze(-2).expand(gram_mat.shape)] = a1[torch.any(mask_zero_2, dim=-1)].mul(
                a1[torch.any(mask_zero_2, dim=-1)]).sum(-1).view(-1)

            angular_kernel = self.angular_weights[..., 0, None, None]
            for p in range(1, self.num_angular_weights):
                angular_kernel = angular_kernel + self.angular_weights[..., p, None, None].mul(gram_mat.pow(p))
        else:
            gram_mat = a1.mul(a2).sum(-1)
            gram_mat[mask_zero_1] = a2[mask_zero_1, :].mul(a2[mask_zero_1, :]).sum(-1)
            gram_mat[mask_zero_2] = a1[mask_zero_2, :].mul(a1[mask_zero_2, :]).sum(-1)

            angular_kernel = self.angular_weights[..., 0, None]
            for p in range(1, self.num_angular_weights):
                angular_kernel = angular_kernel + self.angular_weights[..., p, None].mul(gram_mat.pow(p))

        with gpysettings.lazily_evaluate_kernels(False):
            radial_kernel = self.radial_base_kernel(self.kuma(r1), self.kuma(r2), diag=diag, **params)
        return radial_kernel.mul(angular_kernel)

    def kuma(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(*self.batch_shape, 1, 1)
        beta = self.beta.view(*self.batch_shape, 1, 1)
        radii = (x / self.max_radius).clamp(min=self.eps, max=1)
        res = self.max_radius * (1. - (1. - radii ** alpha + self.eps) ** beta)
        return res

    def num_outputs_per_input(self, x1: torch.Tensor, x2: torch.Tensor) -> int:
        return self.radial_base_kernel.num_outputs_per_input(x1, x2)

    def set_priors(self, priors: Optional[Dict[str, Prior]] = None, with_default=True,
                   without_initial_value=False):
        if priors is None:
            priors = {}
        self.radial_base_kernel.set_priors(priors=priors, with_default=with_default,
                                           without_initial_value=without_initial_value)
        for name in ('angular_weights', 'alpha', 'beta'):
            name_prior = name + '_prior'
            raw_name = 'raw_' + name
            if with_default or name_prior in priors:
                prior = priors.get(name_prior, getattr(settings, name).prior())
                self.register_prior(name_prior, prior, lambda k=name: getattr(self, k),
                                    lambda v, k=name: setattr(self, k, v))
                if without_initial_value:
                    init_value = prior.constraint.initial_value
                    prior.constraint.initial_value = None
                self.register_constraint(raw_name, prior.constraint)
                if without_initial_value:
                    prior.constraint.initial_value = init_value
        return self
