from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch.distributions import Distribution


class SurrogateModel(ABC):
    def __init__(self, *args, train_inputs: Optional[torch.Tensor] = None,
                 train_targets: Optional[torch.Tensor] = None, **kwargs):
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)
        self.train_inputs = train_inputs
        self.train_targets = train_targets

        # observation model, should take a distribution describing
        # the noise-free function and return a distribution that includes
        # the estimated observed noise
        def likelihood(x: Distribution) -> Distribution:
            raise NotImplementedError(f"{self.__class__.__name__}: Must define observation model as 'likelihood'")
        self.likelihood = likelihood
        super().__init__(*args, **kwargs)

    @abstractmethod
    def set_train_data(self, inputs=None, targets=None, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Distribution:
        pass

    # this is optional, only implement if linear updates are needed,
    # there inputs and targets are the new data to be added to the
    # existing train data
    @abstractmethod
    def get_fantasy_model(self, inputs, targets, **kwargs):
        pass
