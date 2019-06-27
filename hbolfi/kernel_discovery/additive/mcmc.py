from typing import Union
from numbers import Number

import torch

from ...models import AdditiveStructureGPModel


class MCMCStructureSelector:
    def __init__(self, model: AdditiveStructureGPModel,
                 seed: Union[int, torch.tensor] = 117,
                 save_trace: bool = False, verbose: bool = False):
        self.current_model = model
        # store state and not generator, because the latter cannot be serialized by pickle/dill
        self.rng_state = torch.Generator().manual_seed(seed).get_state() if isinstance(seed, Number) else seed
        self.save_trace = save_trace
        if save_trace:
            self.trace = []
            self.record(model)
        self.verbose = verbose

    def record(self, model: AdditiveStructureGPModel):
        self.trace.append((model.log_likelihood, model.additive_structure))

    def sample(self) -> AdditiveStructureGPModel:
        raise NotImplementedError

    def run(self, iters: int = 1) -> AdditiveStructureGPModel:
        raise NotImplementedError
