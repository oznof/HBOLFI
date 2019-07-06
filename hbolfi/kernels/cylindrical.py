from typing import Dict, Optional
import math

import torch
import torch.nn
from torch.distributions import Dirichlet

import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from gpytorch import settings as gpysettings

from .. import settings
from ..utils import make_property
from ..kernels import ScaleMaternKernel
from ..distributions import DirichletPrior, TrainingDistribution


# NOTE: differs substantially from the implementation in gpytorch
# NOTE: we assume there is only one point at the origin (no repeats)
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
                 priors: Optional[Dict[str, Prior]] = None,
                 weight_normalization: bool = True,
                 warping: bool = True,
                 **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        if 'active_dims' in kwargs:
            max_radius = float(len(kwargs['active_dims'])) ** 0.5
        self.max_radius = max_radius
        self.num_angular_weights = num_angular_weights
        self.weight_normalization = weight_normalization
        self.warping = warping
        self.eps = eps
        self.radial_base_kernel = radial_base_kernel if radial_base_kernel else ScaleMaternKernel(nu=2.5)
        if priors is None:
            priors = {}
        angular_weights_prior = priors.get('angular_weights_prior', settings.angular_weights.prior())
        if isinstance(angular_weights_prior, DirichletPrior):
            num_angular_weights -= 1  # because it must be in simplex
        self.register_parameter(name="raw_angular_weights",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, num_angular_weights)))
        if not self.warping:
            settings.alpha.set('Delta', torch.ones(1))
            settings.beta.set('Delta', torch.ones(1))
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.set_priors(priors)

    # add attributes: constrained parameters
    angular_weights = make_property('raw_angular_weights')
    alpha = make_property('raw_alpha')
    beta = make_property('raw_beta')

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False,
                xtest: Optional[torch.Tensor] = None, **params) -> torch.Tensor:
        if params.pop('ignore_x2', False):
            x2 = None
        # NOTE: we assume xtest to be a vector (if given), unless x2 is None (used in the original prediction method)
        r1 = torch.zeros(*x1.shape[:-1], 1)
        mask_zero_1 = torch.all(x1 == 0, dim=-1)
        mask_not_zero_1 = ~mask_zero_1
        r1[mask_not_zero_1] = x1[mask_not_zero_1, :].norm(dim=-1, keepdim=True)
        r1[mask_zero_1] = x1[mask_zero_1, :].sum(dim=-1, keepdim=True)
        if x2 is not None:
            r2 = torch.zeros(*x2.shape[:-1], 1)
            mask_zero_2 = torch.all(x2 == 0, dim=-1)
            mask_not_zero_2 = ~mask_zero_2
            r2[mask_not_zero_2] = x2[mask_not_zero_2, :].norm(dim=-1, keepdim=True)
            r2[mask_zero_2] = x2[mask_zero_2, :].sum(dim=-1, keepdim=True)
        else:
            assert(isinstance(xtest, torch.Tensor))
            r2 = torch.ones(xtest.shape[:-1] + (1,)).requires_grad_(True) * 0

        if torch.any(r1 > self.max_radius) or torch.any(r2 > self.max_radius):
            raise RuntimeError(f"Cylindrical kernel not defined for data points with radius > {self.max_radius}")

        atest = None
        if isinstance(xtest, torch.Tensor):
            # TODO: check if its better to detach, in the original implementation it did not detach
            atest = xtest / xtest.norm(dim=-1, keepdim=True)

        a1 = torch.zeros_like(x1)
        a1[mask_not_zero_1, :] = x1[mask_not_zero_1, :] / r1[mask_not_zero_1]
        if torch.any(mask_zero_1):
            a1[mask_zero_1] = 1. / math.sqrt(x1.size(-1)) if xtest is None else atest
        if x2 is not None:
            a2 = torch.zeros_like(x2)
            a2[mask_not_zero_2, :] = x2[mask_not_zero_2, :] / r2[mask_not_zero_2]
            if torch.any(mask_zero_2):
                a2[mask_zero_2] = 1. / math.sqrt(x2.size(-1)) if xtest is None else atest
        else:
            a2 = atest

        angular_weights = self.angular_weights
        if self.weight_normalization and not isinstance(self.angular_weights_prior, Dirichlet):
            angular_weights = angular_weights / angular_weights.sum(dim=-1, keepdim=True)
        if not diag:
            gram_mat = a1.matmul(a2.transpose(-2, -1))
            if atest is None:
                gram_mat[mask_zero_1] = a2[torch.any(mask_zero_1, dim=-1)].mul(a2[torch.any(mask_zero_1,
                                                                                            dim=-1)]).sum(-1)
                gram_mat[mask_zero_2.unsqueeze(-2).expand(gram_mat.shape)] = a1[torch.any(mask_zero_2, dim=-1)].mul(
                    a1[torch.any(mask_zero_2, dim=-1)]).sum(-1).view(-1)

            angular_kernel = angular_weights[..., 0, None, None]
            for p in range(1, self.num_angular_weights):
                angular_kernel = angular_kernel + angular_weights[..., p, None, None].mul(gram_mat.pow(p))
        else:
            gram_mat = a1.mul(a2).sum(-1)
            if atest is None:
                gram_mat[mask_zero_1] = a2[mask_zero_1, :].mul(a2[mask_zero_1, :]).sum(-1)
                gram_mat[mask_zero_2] = a1[mask_zero_2, :].mul(a1[mask_zero_2, :]).sum(-1)

            angular_kernel = angular_weights[..., 0, None]
            for p in range(1, self.num_angular_weights):
                angular_kernel = angular_kernel + angular_weights[..., p, None].mul(gram_mat.pow(p))

        with gpysettings.lazily_evaluate_kernels(False):
            if self.warping:
                r1 = self.kuma(r1)
                r2 = self.kuma(r2)
            radial_kernel = self.radial_base_kernel(r1, r2, diag=diag, **params)
        return radial_kernel.mul(angular_kernel)

    def kuma(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(*self.batch_shape, 1, 1)
        beta = self.beta.view(*self.batch_shape, 1, 1)
        radii = (x / self.max_radius).clamp(min=0, max=1)
        res = self.max_radius * (1. - (1. - radii ** alpha) ** beta)
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


# extended to compute covariances
def original_prediction_strategy_cylindrical(model, x_pred: torch.Tensor):
    # NOTE: assuming x_train and x_pred to be matrices, just like in the original method (does not support batches)
    # does not compute covariances
    if not isinstance(model.prediction_strategy, tuple):
        # use prediction_strategy to store L_not0_not0, and other variables related to the training data
        # every time, set_data is called this attributes will be set to None, allowing these variables to be
        # cached
        # TODO: add support to linear updates with get_fantasy_model
        mask_train_0 = torch.all(model.train_inputs[0] == 0, dim=-1)
        mask_train_not0 = ~mask_train_0
        x_train = model.train_inputs[0]
        if model.to_centered_cube_transform is not None:
            x_train = model.to_centered_cube_transform(x_train)
        x_train_not0 = x_train[mask_train_not0]
        x_train_0 = x_train[mask_train_0]
        eye_mat = torch.eye(x_train_not0.size(-2))
        gram_mat = model.kernel(x_train_not0, x_train_not0) + model.likelihood.noise * eye_mat
        chol_jitter = 0
        while True:
            try:
                L_not0_not0 = torch.cholesky(gram_mat + eye_mat * chol_jitter, upper=False)
                break
            except RuntimeError:
                chol_jitter = gram_mat.data[0, 0] * 1e-6 if chol_jitter == 0 else chol_jitter * 10
        y_delta = model.train_targets - model.mean(x_train)
        y_delta_not0 = y_delta[mask_train_not0]
        y_delta_0 = y_delta[mask_train_0]
        model.prediction_strategy = (x_train_not0, x_train_0, L_not0_not0, chol_jitter,
                                     y_delta, y_delta_not0, y_delta_0)
    (x_train_not0, x_train_0, L_not0_not0, chol_jitter, y_delta, y_delta_not0, y_delta_0) = model.prediction_strategy

    n_pred, dim = x_pred.shape
    x_pred_radius = torch.sqrt(torch.sum(x_pred ** 2, 1, keepdim=True))
    assert((x_pred_radius.detach() > 0).all())

    if model.to_centered_cube_transform is not None:
        x_pred = model.to_centered_cube_transform(x_pred)

    with gpytorch.settings.lazily_evaluate_kernels(False):
        K_0_0 = model.kernel(x1=x_train_0, x2=x_train_0)
        K_not0_0 = model.kernel(x1=x_train_not0, xtest=x_pred, ignore_x2=True)
        K_not0_pre = model.kernel(x1=x_train_not0, x2=x_pred)
        K_pre_0 = model.kernel(x1=x_pred, x2=torch.zeros(1, dim))
    K_pre_pre = model.kernel(x1=x_pred, x2=x_pred)

    chol_solver = torch.solve(torch.cat([K_not0_0, K_not0_pre, y_delta_not0[:, None]], 1), L_not0_not0)[0]
    chol_solver_q = chol_solver[:, :n_pred]  # L_10 = L_11^-1 K_10  (n_1 x n_pred), expanding origin to n_pred
    chol_solver_k = chol_solver[:, n_pred:n_pred * 2]  # L_11^-1 K_1*  (n_1 x n_pred)
    chol_solver_y = chol_solver[:, n_pred * 2:n_pred * 2 + 1]  # L_11^-1 y1  (n_1 x 1)
    # minor detail: chol_jitter should be squared here, but the original implementation didn't have it
    # also, it is implicitly assumed that there is only one point in the origin
    # this condition is not checked, nor enforced
    sol_p_sqr = K_0_0 + model.likelihood.noise + chol_jitter - (chol_solver_q ** 2).sum(0).view(-1, 1)
    sol_p = torch.sqrt(sol_p_sqr.clamp(min=1e-12))  # L_00  normally (1 x 1), but expanded to (n_pred, 1)
    # sol_k_bar = K_*0 L_00^-1 + K_*1 L_10^-1
    # with L_10^-1 = - L_11^-T L_10 L_00^-T
    # equivalent to (less efficient):
    # sol_k_bar = K_pre_0 / sol_p
    # for i in range(n_pred):
    #    sol_k_bar[i] -= chol_solver_k.t()[i, :] @ chol_solver_q[:, i] / sol_p[i]
    # or sol_k_bar3 = K_pre_0 / sol_p - torch.diag(chol_solver_k.t() @ chol_solver_q).view(-1, 1) / sol_p
    sol_k_bar = (K_pre_0 - (chol_solver_q * chol_solver_k).sum(0).view(-1, 1)) / sol_p
    # rhs solve: L_00^-1 y0 - L_00^-1 L_01 L_11^-1 y1
    sol_y_bar = (y_delta_0 - torch.mm(chol_solver_q.t(), chol_solver_y)) / sol_p
    # pred_mean = mean_** + K_*1 L_11^-T L_11^-1 y1 + terms involving 0
    pred_mean = model.mean(x_pred) + (torch.mm(chol_solver_k.t(), chol_solver_y) + sol_k_bar * sol_y_bar).squeeze()
    # pred_var = K_pre_pre_diag - ((chol_solver_k ** 2).sum(0).view(-1, 1) + sol_k_bar ** 2).squeeze()
    # K_** - K_*1 L_11^-T L_11 ^-1 K_1* -  (K_*0 L_00^-1 + K_*1 L_10^-1) ( K_*0 L_00^-1 + K_*1 L_10^-1)^T
    pred_covar = K_pre_pre - chol_solver_k.t().mm(chol_solver_k) - sol_k_bar.mm(sol_k_bar.t())
    return gpytorch.distributions.MultivariateNormal(pred_mean, pred_covar)


def original_training_strategy_cylindrical(model, x):
    assert(torch.equal(x, model.train_inputs[0]))

    def nll(y):
        assert(torch.equal(y, model.train_targets))
        return _original_training_strategy_cylindrical(model, x, y, save_cache=True)
    return TrainingDistribution(nll)


def _original_training_strategy_cylindrical(model, x_train, y_train, save_cache=False):
    mask_train_0 = torch.all(x_train == 0, dim=-1)
    mask_train_not0 = ~mask_train_0
    if model.to_centered_cube_transform is not None:
        x_train = model.to_centered_cube_transform(x_train)
    x_train_not0 = x_train[mask_train_not0]
    x_train_0 = x_train[mask_train_0]
    eye_mat = torch.eye(x_train_not0.size(-2))
    gram_mat = model.kernel(x_train_not0, x_train_not0) + model.likelihood.noise * eye_mat
    chol_jitter = 0
    while True:
        try:
            L_not0_not0 = torch.cholesky(gram_mat + eye_mat * chol_jitter, upper=False)
            break
        except RuntimeError:
            chol_jitter = gram_mat.data[0, 0] * 1e-6 if chol_jitter == 0 else chol_jitter * 10
    y_delta = y_train - model.mean(x_train)
    y_delta_not0 = y_delta[mask_train_not0]
    y_delta_0 = y_delta[mask_train_0]
    if save_cache:
        model.prediction_strategy = (x_train_not0, x_train_0, L_not0_not0, chol_jitter,
                                     y_delta, y_delta_not0, y_delta_0)

    with gpytorch.settings.lazily_evaluate_kernels(False):
        K_0_0 = model.kernel(x1=x_train_0, x2=x_train_0)
        K_not0_0 = model.kernel(x1=x_train_not0, xtest=x_train_not0, ignore_x2=True)

    chol_solver = torch.solve(torch.cat([y_delta_not0[:, None], K_not0_0], 1), L_not0_not0)[0]
    chol_solver_y = chol_solver[:, :1]  # L_11^-1 y1  (n_1 x 1)
    chol_solver_q = chol_solver[:, 1:]  # L_10 = L_11^-1 K_10  (n_1 x n_1)
    sol_p_sqr = K_0_0 + model.likelihood.noise + chol_jitter - (chol_solver_q ** 2).sum(0).view(-1, 1)
    sol_p = torch.sqrt(sol_p_sqr.clamp(min=1e-12))  # L_00  normally (1 x 1), but expanded to (n_1, 1)
    # solve related to the origin.
    # L_00^-1 y_0 - L_00^-1 L_01 L_11^-1 y1
    # since L_00, L_01 have been expanded -> (n1 x 1)
    # these are then averaged out in nll
    sol_y_i = (y_delta_0 - chol_solver_q.t().mm(chol_solver_y)) / sol_p

    nll = 0.5 * (torch.sum(chol_solver_y ** 2) + torch.mean(sol_y_i ** 2)) + torch.sum(
        torch.log(torch.diag(L_not0_not0))) + torch.mean(torch.log(sol_p)) + 0.5 * x_train.size(0) *\
        math.log(2 * math.pi)
    return nll