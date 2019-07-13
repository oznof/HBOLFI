from typing import Union, Optional

import torch
from torch.distributions import constraints, StudentT
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform


class LogStudentT(TransformedDistribution):
    r"""
    Log Student t-distribution parameterized by `loc`, `scale` and `df`  where:
        X ~ StudentT(loc, scale, df)
        Y = exp(X) ~ LogStudentT(loc, scale, df)
    """

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'df': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc: Union[float, torch.Tensor], scale: Union[float, torch.Tensor],
                 df: Union[float, torch.Tensor], validate_args: Optional[bool] = None):
        base_dist = StudentT(loc=loc, scale=scale, df=df, validate_args=validate_args)
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape: torch.Size, _instance=None):
        new = self._get_checked_instance(LogStudentT, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def df(self):
        return self.base_dist.df
