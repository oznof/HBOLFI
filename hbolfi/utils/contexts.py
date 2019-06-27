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
            torch.set_rng_state(self.state)
        return self

    def __exit__(self, *args):
        if self.generator_or_state is not None:
            current_rng_state = torch.get_rng_state()
            if self.isgenerator:
                self.generator_or_state.set_state(current_rng_state)
            else:
                self.generator_or_state = current_rng_state
            torch.set_rng_state(self.torch_state)


# @contextmanager
# def rng(generator_or_state):
#     if generator_or_state is not None:
#         if isinstance(generator_or_state, torch.Tensor):
#             isgenerator = False
#             state = generator_or_state
#         else:
#             isgenerator = True
#             state = generator_or_state.get_state()
#         torch_state = torch.get_rng_state()
#         torch.set_rng_state(state)
#     yield
#     if generator_or_state is not None:
#         current_rng_state = torch.get_rng_state()
#         if isgenerator:
#             generator_or_state.set_state(current_rng_state)
#         else:
#             generator_or_state = current_rng_state
#         torch.set_rng_state(torch_state)
