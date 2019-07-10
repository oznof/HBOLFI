#!/usr/bin/env python3
from typing import Dict, Optional

import torch
from torch.nn import ModuleList
from gpytorch.means import Mean
from gpytorch.priors import Prior

from .. import settings
from ..utils import make_property


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
        self.set_priors(priors)

    a = make_property('raw_a')
    b = make_property('raw_b')
    c = make_property('raw_c')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_dims is None:
            m_a = x.pow(2).matmul(self.a.unsqueeze(-1))  # b x n x 1
            m_b = x.matmul(self.b.unsqueeze(-1))
            return (m_a + m_b).squeeze() + self.c  # b x n, broadcast c along n
        else:
            # make sure to set self.num_groups
            xi = x[..., self.active_dims]
            m_a = xi.pow(2).matmul(self.a[..., self.active_dims].unsqueeze(-1))
            m_b = xi.matmul(self.b[..., self.active_dims].unsqueeze(-1))
            return (m_a + m_b).squeeze() + self.c / self.num_groups

    def set_priors(self, priors: Optional[Dict[str, Prior]] = None, with_default=True,
                   without_initial_value=False):
        if priors is None:
            priors = {}
        for name in ('a', 'b', 'c'):
            name_prior = name + '_prior'
            raw_name = 'raw_' + name
            if with_default or 'quadpoly_' + name_prior in priors:
                prior = priors.get('quadpoly_' + name_prior, getattr(settings, 'quadpoly_' + name).prior())
                self.register_prior(name_prior, prior, lambda k=name: getattr(self, k),
                                    lambda v, k=name: setattr(self, k, v))
                if without_initial_value:
                    init_value = prior.constraint.initial_value
                    prior.constraint.initial_value = None
                self.register_constraint(raw_name, prior.constraint)
                if without_initial_value:
                    prior.constraint.initial_value = init_value
