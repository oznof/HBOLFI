import torch

from .acquisition_base import AcquisitionFunction
from ..distributions import normal_prob, normal_cdf


class EI(AcquisitionFunction):

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.dim)
        pred_mvn = self.model(x)
        mean = pred_mvn.mean
        var = pred_mvn.variance
        std = torch.sqrt(var).clamp(1e-20)
        ref = torch.min(self.model.train_targets)
        z = (ref - mean) / std
        return (std * normal_prob(z) + (ref - mean) * normal_cdf(z)).clamp(0).view(-1, x.size(0)).mean(0)
