from typing import Callable
import torch


class TrainingDistribution(torch.distributions.Distribution):
    arg_constraints = {}

    def __init__(self, nll: Callable[[torch.Tensor], torch.Tensor]):
        self._nll = nll

    def log_prob(self, targets):
        return -self._nll(targets)
