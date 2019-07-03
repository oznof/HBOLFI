import math

from sympy.solvers import solve
from sympy import Symbol

import numpy as np
from scipy.special import gammaincc
from scipy.optimize import minimize

import matplotlib.pyplot as plt

import torch
from torch.nn.functional import softplus
from torch.distributions import Transform, constraints

from .inverse_gamma import InvGamma


def normal_prob(value):
    return 1 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * value.pow(2))


def normal_log_prob(value):
    return -0.5 * (math.log(2 * math.pi) + value.pow(2))


def normal_cdf(value):
    return 0.5 * (1 + torch.erf(value / math.sqrt(2)))


def inverse_normal_cdf(value):
    return torch.erfinv(2 * value - 1) * math.sqrt(2)


def inv_gamma_cdf(x, a, b):
    return gammaincc(a, b / x)


def invgamma_solver(l, u, method='minimize', verbose=False, show_plot=False):
    if method == 'minimize':
        def fun(p):
            lower_q = inv_gamma_cdf(l, *p)
            upper_q = inv_gamma_cdf(u, *p)
            return (lower_q - 0.01) ** 2 + (upper_q - 0.99) ** 2

        res = minimize(fun=fun, x0=np.ones(2), method='L-BFGS-B', jac=False, bounds=[(0, np.inf), (0, np.inf)])
        a, b = res['x']
    elif method == 'normal':
        a = Symbol("a")
        b = ((u - l) / 6 * (a - 1)**2 * (a - 2)) ** 0.5
        res = solve(b / (a - 1) + 3 * b**2 / ((a - 1)**2 * (a - 2)) - u, a)
        a = float(res[0])
        b = ((u - l) / 6 * (a - 1)**2 * (a - 2)) ** 0.5
    else:
        raise NotImplementedError("Unknown method for invgamma_solver. Either 'minimize' or 'normal")
    if verbose:
        print(inv_gamma_cdf(l, a, b))
        print(inv_gamma_cdf(u, a, b))
    if show_plot:
        x = torch.linspace(0.01, u+1, 1000)
        y = InvGamma(a, b).log_prob(x).exp().numpy()
        plt.plot(x, y)
        plt.show()
    return a, b


class SoftPlusTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \log(1 + \exp(x))`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SoftPlusTransform)

    def _call(self, x):
        return softplus(x)

    def _inverse(self, y):
        return (y.exp() - 1).log()

    def log_abs_det_jacobian(self, x, y):
        return -torch.log(1 + torch.exp(-x))
