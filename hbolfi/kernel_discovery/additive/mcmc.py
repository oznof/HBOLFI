from typing import Optional, Union
from numbers import Number

import torch

from ...models import AdditiveStructureGPModel


class MCMCStructureSelector:
    def __init__(self, model: AdditiveStructureGPModel,
                 seed: Optional[Union[int, torch.tensor]] = None,
                 save_trace: bool = False, verbose: bool = False):
        self.current_model = model
        # store state and not generator, because the latter cannot be serialized by pickle/dill
        if seed is None:
            seed = torch.get_rng_state()
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

    def run(self, model: Optional[AdditiveStructureGPModel] = None, iters: int = 1) -> AdditiveStructureGPModel:
        raise NotImplementedError
