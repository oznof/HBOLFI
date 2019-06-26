from typing import Union, Optional
import math

import torch
from torch.distributions import constraints, StudentT
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform


class HalfStudentT(TransformedDistribution):
    r"""
    Half Student t-distribution parameterized by `scale` and `df`  where:
        X ~ StudentT(0, scale, df)
        Y = |X| ~ HalfStudentT(scale, df)
    """

    arg_constraints = {'df': constraints.positive, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, scale: Union[float, torch.Tensor], df: Union[float, torch.Tensor],
                 validate_args: Optional[bool] = None):
        base_dist = StudentT(loc=0, scale=scale, df=df, validate_args=validate_args)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape: torch.Size, _instance=None):
        new = self._get_checked_instance(HalfStudentT, _instance)
        return super().expand(batch_shape, _instance=new)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob[value.expand(log_prob.shape) < 0] = -float('inf')
        return log_prob

    def cdf(self, value):
        cdf = 2 * self.base_dist.cdf(value) - 1
        cdf[value.expand(cdf.shape) < 0] = 0.
        return cdf

    def icdf(self, prob):
        return self.base_dist.icdf((prob + 1) / 2)

    @property
    def df(self):
        return self.base_dist.df

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        m = self.loc.clone()
        idx = self.df > 1
        m[idx] = 2 * self.scale[idx] * torch.sqrt(self.df[idx] / math.pi) * (1 /  (self.df[idx] - 1) *
                 torch.exp(torch.lgamma((self.df[idx] + 1) / 2.) - torch.lgamma(self.df[idx] / 2)))
        m[1 - idx] = float('nan')
        return m

    @property
    def variance(self):
        v = self.df.clone()
        idx = self.df > 2
        df = self.df[idx]
        v[idx] = self.scale[idx] ** 2 * (df / (df - 2) - 4 * df / (math.pi * (df - 1) ** 2) *
                                         torch.exp(torch.lgamma((df + 1) / 2.) - torch.lgamma(df / 2.)) ** 2)
        v[1 - idx] = float('nan')
        return v
