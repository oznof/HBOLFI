#!/usr/bin/env python3
from typing import Dict, Optional
from functools import partial

import torch
from torch.nn import ModuleList
from gpytorch.means import Mean
from gpytorch.constraints import Positive
from gpytorch.priors import Prior

DEFAULT_CONSTRAINTS = {'a': Positive}


class QuadPolyMean(Mean):
    def __init__(self,
                 input_dim:       int,
                 active_dims:     Optional[torch.Tensor] = None,
                 priors:          Optional[Dict[str, Prior]] = None,
                 batch_shape:     torch.Size = torch.Size(),
                 num_groups:      Optional[int] = None):
        super().__init__()
        self.active_dims = active_dims
        self.batch_shape = batch_shape
        self.num_groups = num_groups
        self.register_parameter(name='raw_a', parameter=torch.nn.Parameter(torch.zeros(*batch_shape, input_dim)))
        self.register_parameter(name='raw_b', parameter=torch.nn.Parameter(torch.zeros(*batch_shape, input_dim)))
        self.register_parameter(name='raw_c', parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if priors is None:
            priors = {}
        for raw_p, _ in self.named_parameters():
            p = raw_p.replace('raw_','')
            prior_name = p + '_prior'
            prior = priors.get(prior_name)
            if prior:  # register prior on constrained parameter
                self.register_prior(prior_name, prior, lambda p=p: getattr(self, p),
                                    lambda v, k=raw_p: self._setter_p(v, k))
                self.register_constraint(raw_p, prior.support)
            else:
                constraint = DEFAULT_CONSTRAINTS.get(p)
                if constraint:
                    self.register_constraint(raw_p, constraint())
        self.initialize(a=1., b=0.)

    def _getter_p(self, k: str) -> torch.Tensor:  # unconstrained -> constrained
        constraint = self._constraints.get(f"{k}_constraint")
        return constraint.transform(getattr(self, k)) if constraint else getattr(self, k)

    def _setter_p(self, v, k: str):  # constrained -> unconstrained
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(getattr(self, k))
        constraint = self._constraints.get(f"{k}_constraint")
        self.initialize(**{k: constraint.inverse_transform(v) if constraint else v})

    # add attributes: constrained parameters
    # add attributes: constrained parameters
    a = property(partial(_getter_p, k='raw_a'), partial(_setter_p, k='raw_a'))
    b = property(partial(_getter_p, k='raw_b'), partial(_setter_p, k='raw_b'))
    c = property(partial(_getter_p, k='raw_c'), partial(_setter_p, k='raw_c'))
    # a = property(lambda self, k='raw_a': self._getter_p(k), lambda self, v, k='raw_a': self._setter_p(v, k))
    # b = property(lambda self, k='raw_b': self._getter_p(k), lambda self, v, k='raw_b': self._setter_p(v, k))
    # c = property(lambda self, k='raw_c': self._getter_p(k), lambda self, v, k='raw_c': self._setter_p(v, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_dims is None:
            A = x.pow(2).matmul(self.a.unsqueeze(-1))  # b x n x 1
            B = x.matmul(self.b.unsqueeze(-1))
            return (A + B).squeeze() + self.c  # b x n, broadcast c along n
        else:
            # make sure to set self.num_groups
            xi = x[..., self.active_dims]
            A = xi.pow(2).matmul(self.a[..., self.active_dims].unsqueeze(-1))
            B = xi.matmul(self.b[..., self.active_dims].unsqueeze(-1))
            return (A + B).squeeze() + self.c / self.num_groups


if __name__ == '__main__':
    from hbolfi.distributions import LogNormalPrior, InvGammaPrior
    mean = QuadPolyMean(4, priors={'a_prior': LogNormalPrior(torch.ones(4) * 1.5, torch.ones(4) * 3.)})
    x0 = torch.arange(3*4, dtype=torch.float).reshape((3, 4))
    x1 = torch.clone(x0) + 20
    x2 = torch.stack((x0, x1))
    print(x2)
    print(x2.shape)
    print(mean(x2))
    print("===========")
    print(mean(x0))
    print("=============")
    print(mean(x1))
    print("=============")
    print(torch.sum(x2 ** 2, dim=-1))
