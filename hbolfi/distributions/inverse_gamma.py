from typing import Union, Optional

import torch
from torch.distributions import constraints, Gamma
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import PowerTransform
from torch.distributions.utils import broadcast_all


class InvGamma(TransformedDistribution):
    """InvGamma parameterized by alpha and beta
          pdf(x) = beta^alpha / Gamma(alpha) * x^(-alpha-1) * exp(-beta / x)
          where alpha > 0 and beta > 0.
    """
    arg_constraints = {'alpha': constraints.positive,
                       'beta': constraints.positive}
    support = constraints.positive

    def __init__(self, alpha: Union[float, torch.Tensor], beta: Union[float, torch.Tensor],
                 validate_args: Optional[bool] = None):
        self.alpha, self.beta = broadcast_all(alpha, beta)
        base_dist = Gamma(concentration=self.alpha, rate=self.beta)
        super().__init__(base_dist, PowerTransform(exponent=-1), validate_args=validate_args)

    def expand(self, batch_shape: torch.Size, _instance=None):
        new = self._get_checked_instance(InvGamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        new.beta = self.beta.expand(batch_shape)
        base_dist = self.base_dist.expand(batch_shape)
        transform = PowerTransform(exponent=-1)
        super(InvGamma, new).__init__(base_dist, transform, validate_args=False)
        new._validate_args = self._validate_args
        return new
