from typing import Optional, Tuple, Union
import math

import torch

from .acquisition_base import AcquisitionFunction
from .additive import AdditiveAcquisitionFunction
from ..models import AdditiveStructureGPModel


class LCB(AcquisitionFunction):
    def __init__(self, *args, beta: Optional[Union[float, torch.Tensor]] = None,
                 delta: Optional[float] = 0.1, **kwargs):
        assert(0 < delta < 1)
        self.delta = delta
        self._beta = beta
        super().__init__(*args, **kwargs)

    @property
    def beta(self) -> torch.tensor:
        if self._beta is not None:
            return torch.as_tensor(self._beta)
        return 2 * ((2 * self.dim + 2) * torch.log(self.t) + math.log(math.pi ** 2 / (3 * self.delta)))

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.dim)
        pred_mvn = self.model(x)
        mean = pred_mvn.mean
        var = pred_mvn.variance * self.beta
        # print("IN EVALUATE")
        # print("x: ", x)
        # print("mean: ", mean)
        # print("var: ", var)
        return (mean - var.sqrt()).view(-1, x.size(0)).mean(0)


# AdditiveAcquisitionFunction is added first, so that it calls
# AdditiveAcquisitionFunction.evaluate -> LCB.evaluate
# AdditiveAcquisitionFunction.optimize -> LCB.optimize (which is AcquisitionFunction.optimize)
class LCBAdd(AdditiveAcquisitionFunction, LCB):
    """LCB for Additive GP"""
    def __init__(self, model: AdditiveStructureGPModel, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.max_d = max([len(pi) for pi in self.model.additive_structure])

    @property
    def beta(self) -> torch.Tensor:
        if self._beta is not None:
            return torch.as_tensor(self._beta)
        return 0.2 * self.max_d * torch.log(2 * self.t)

    def optimize(self, x0: torch.Tensor, retain_graph: bool = False) -> Tuple[torch.Tensor, float]:
        self.max_d = max([len(pi) for pi in self.model.additive_structure])
        return super().optimize(x0, retain_graph=retain_graph)
