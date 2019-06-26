from typing import Dict, Optional
from functools import partial

import torch
import torch.nn

from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from gpytorch import settings
from gpytorch.constraints import Positive

from hbolfi import settings
from ..kernels import ScaleMaternKernel


# based to some extent on the implementation in gpytorch.kernels.CylindricalKernel
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
                 angular_weights_prior: Optional[Prior] = None,
                 alpha_prior: Optional[Prior] = None, beta_prior: Optional[Prior] = None, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.max_radius = max_radius
        self.num_angular_weights = num_angular_weights
        self.eps = eps
        if radial_base_kernel is None:
            lengthscale_prior = settings.lengthscale.prior('Uniform')
            outputscale_prior = settings.outputscale.prior('Uniform')
            radial_base_kernel = ScaleMaternKernel(nu=2.5, lengthscale_prior=lengthscale_prior,
                                                   lengthscale_constraint=lengthscale_prior.constraint,
                                                   outputscale_prior=outputscale_prior,
                                                   outputscale_constraint=outputscale_prior.constraint)
        self.radial_base_kernel = radial_base_kernel
        if angular_weights_prior is None:
            angular_weights_prior = settings.angular_weights.prior('Dirichlet')
            num_angular_weights -= 1  # because it must be in simplex
        if alpha_prior is None:
            alpha_prior = settings.alpha.prior('SpikeUniformSlab')
        if beta_prior is None:
            beta_prior = settings.beta.prior('SpikeUniformSlab')
        self.register_parameter(name="raw_angular_weights",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, num_angular_weights)))
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        priors = {'angular_weights_prior': angular_weights_prior, 'alpha_prior': alpha_prior, 'beta_prior': beta_prior}
        for raw_p, _ in self.named_parameters():
            if '.' in raw_p:
                continue
            p = raw_p.replace('raw_', '')
            prior_name = p + '_prior'
            prior = priors.get(prior_name)
            if prior:
                self.register_prior(prior_name, prior, lambda k=p: getattr(self, k),
                                    lambda v, k=raw_p: self._setter_p(v, k))
            self.register_constraint(raw_p, prior.constraint if prior else Positive())

    def _getter_p(self, k: str) -> torch.Tensor:  # unconstrained -> constrained
        constraint = self._constraints.get(f"{k}_constraint")
        return constraint.transform(getattr(self, k)) if constraint else getattr(self, k)

    def _setter_p(self, v, k: str):  # constrained -> unconstrained
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(getattr(self, k))
        constraint = self._constraints.get(f"{k}_constraint")
        self.initialize(**{k: constraint.inverse_transform(v) if constraint else v})

    # add attributes: constrained parameters
    angular_weights = property(partial(_getter_p, k='raw_angular_weights'), partial(_setter_p, k='raw_angular_weights'))
    alpha = property(partial(_getter_p, k='raw_alpha'), partial(_setter_p, k='raw_alpha'))
    beta = property(partial(_getter_p, k='raw_beta'), partial(_setter_p, k='raw_beta'))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        x1_, x2_ = x1.clone(), x2.clone()
        # Jitter datapoints that are exactly 0
        x1_[torch.all(x1_ == 0, dim=-1)] += self.eps
        x2_[torch.all(x2_ == 0, dim=-1)] += self.eps
        r1 = x1_.norm(dim=-1, keepdim=True)
        r2 = x2_.norm(dim=-1, keepdim=True)

        if torch.any(r1 > self.max_radius) or torch.any(r2 > self.max_radius):
            raise RuntimeError(f"Cylindrical kernel not defined for data points with radius > {self.max_radius}")

        a1, a2 = x1.div(r1), x2.div(r2)
        if not diag:
            gram_mat = a1.matmul(a2.transpose(-2, -1))
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = self.angular_weights[..., 0, None, None]
                else:
                    angular_kernel = angular_kernel + self.angular_weights[..., p, None, None].mul(gram_mat.pow(p))
        else:
            gram_mat = a1.mul(a2).sum(-1)
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = self.angular_weights[..., 0, None]
                else:
                    angular_kernel = angular_kernel + self.angular_weights[..., p, None].mul(gram_mat.pow(p))

        with settings.lazily_evaluate_kernels(False):
            radial_kernel = self.radial_base_kernel(self.kuma(r1), self.kuma(r2), diag=diag, **params)
        return radial_kernel.mul(angular_kernel)

    def kuma(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(*self.batch_shape, 1, 1)
        beta = self.beta.view(*self.batch_shape, 1, 1)
        res = self.max_radius * (1 - (1 - x.div(self.max_radius).clamp(min=0, max=1).pow(alpha) + self.eps).pow(beta))
        return res

    def num_outputs_per_input(self, x1: torch.Tensor, x2: torch.Tensor) -> int:
        return self.radial_base_kernel.num_outputs_per_input(x1, x2)

    def set_priors(self, priors: Optional[Dict[str, Prior]] = None, with_default=True):
        if priors is None:
            priors = {}
        self.radial_base_kernel.set_priors(priors=priors, with_default=with_default)
        for name in ('angular_weights', 'alpha', 'beta'):
            name_prior = name + '_prior'
            raw_name = 'raw_' + name
            if with_default or name_prior in priors:
                prior = priors.get(name_prior, getattr(settings, name).prior())
                self.register_prior(name_prior, prior, lambda k=name: getattr(self, k),
                                    lambda v, k=raw_name: self._setter_p(v, k))
                self.register_constraint(raw_name, prior.constraint if prior else Positive())
