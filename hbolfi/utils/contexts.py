import numpy as np
import torch


class rng(object):
    def __init__(self, generator_or_state):
        self.generator_or_state = generator_or_state
        if self.generator_or_state is not None:
            if isinstance(self.generator_or_state, torch.Tensor):
                self.isgenerator = False
                self.state = self.generator_or_state
            else:
                self.isgenerator = True
                self.state = generator_or_state.get_state()

    def __enter__(self):
        if self.generator_or_state is not None:
            self.torch_state = torch.get_rng_state()
            self.numpy_state = np.random.get_state()
            torch.set_rng_state(self.state)
            np.random.seed(self.state.numpy())
        return self

    def __exit__(self, *args):
        if self.generator_or_state is not None:
            current_rng_state = torch.get_rng_state()
            if self.isgenerator:
                self.generator_or_state.set_state(current_rng_state)
            else:
                self.generator_or_state = current_rng_state
            torch.set_rng_state(self.torch_state)
            np.random.set_state(self.numpy_state)
