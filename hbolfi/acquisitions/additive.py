from typing import Tuple

import torch
from torch.distributions import AffineTransform, biject_to, constraints

from .acquisition_base import AcquisitionFunction
from ..models import AdditiveStructureGPModel


# for now, this only works if the search_space is a hypercube specified by 'bounds'
class AdditiveAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: AdditiveStructureGPModel, *args, **kwargs):
        assert (isinstance(model, AdditiveStructureGPModel))
        super().__init__(model, *args, **kwargs)
        self.active_dims = torch.arange(self.dim)
        assert('bounds' in self.search_space)
        self._cached_transforms = {}

    def to_transforms(self, bounds):
        if bounds in self._cached_transforms:
            return self._cached_transforms[bounds]
        transforms = {'from_cube': AffineTransform(bounds[0], bounds[1] - bounds[0], 0),
                      'from_real': biject_to(constraints.interval(bounds[0], bounds[1]))}
        self._cached_transforms[bounds] = transforms
        return transforms

    def optimize(self, x0: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        The acquisition function decomposes additively over the partition.
        The arg min is given by the union of the arg mins.
        """
        x0 = x0.view(1, -1)
        loc = torch.zeros_like(x0)
        val = 0
        partition = self.model.additive_structure
        transforms = self.transforms
        for i, active_dims in enumerate(partition):
            self.active_dims = active_dims
            bounds = self.search_space['bounds'][:, active_dims]
            self.transforms = self.to_transforms(bounds)
            # restrict model to the ith group
            # it still expects inputs of the same shape, slicing internally
            # so there is the need to augment the input before evaluation
            self.model.restrict(i)
            loc[0, active_dims], v = super().optimize(x0[:, active_dims])
            val += v
        # restore original model, bounds, active_dims
        self.model.unrestrict()
        self.transforms = transforms
        self.active_dims = torch.arange(self.dim)
        return loc, val

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x_augmented = torch.zeros(x.size(0), self.dim)
        x_augmented[:, self.active_dims] = x
        return super().evaluate(x_augmented)

    def acquire(self, *args, **kwargs):
        self._cached_transforms = {}
        return super().acquire(*args, **kwargs)