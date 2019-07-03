import torch
from torch.distributions import constraints
from torch.distributions.transforms import AffineTransform
from torch.distributions import HalfNormal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.utils import broadcast_all


class ShiftedHalfNormal(TransformedDistribution):
    r"""
    Creates a shifted half-normal distribution parameterized by `loc` and `scale` where::

        X ~ Normal(0, scale)
        Y = |X| + loc ~ ShiftedHalfNormal(loc, scale)

    Args:
        loc (float or Tensor): location / shift
        scale (float or Tensor): scale of the full Normal distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        base_dist = HalfNormal(scale)
        super().__init__(base_dist, [AffineTransform(loc=self.loc, scale=1.)],
                         validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.greater_than_eq(self.loc)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ShiftedHalfNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)
