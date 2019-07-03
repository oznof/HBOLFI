import math
import torch

# Functions adapted from
# https://github.com/ChangYong-Oh/HyperSphere/blob/master/HyperSphere/test_functions/benchmarks.py
# To be evaluated on [-1, 1]^d


class Branin:
    """
    Input is scaled and shifted to that odd dimensions are in [-5, 10] and even dimensions in [0, 15]
    Additive with non-overlapping groups, each group has 2 elements (in order)
    """
    has_multiple_minima = True

    def __init__(self, ndim: int):
        assert (ndim % 2 == 0)
        self.ndim = ndim
        self.n_repeat = ndim // 2
        self.shift = torch.tensor([2.5, 7.5]).repeat(self.n_repeat)
        self.parameters = (1, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi, 6, 10, 1.0 / (8 * math.pi))
        min_loc = [torch.tensor([-math.pi, 12.275]), torch.tensor([math.pi, 2.275]), torch.tensor([9.42478, 2.475])]
        min_loc = [(l - torch.tensor([2.5, 7.5])) / 7.5 for l in min_loc]
        self.min_loc = [l.repeat(self.n_repeat) for l in min_loc]
        self.min_val = self.eval(self.min_loc[0]).item()

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.ndim)
        x = x * 7.5 + self.shift
        a, b, c, r, s, t = self.parameters
        output = torch.tensor(0.)
        for i in range(self.n_repeat):
            output += a * (x[:, 2 * i + 1] - b * x[:, 2 * i] ** 2 + c * x[:, 2 * i] - r) ** 2 + \
                      s * (1 - t) * torch.cos(x[:, 2 * i]) + s
        output /= self.n_repeat
        return output


class Hartmann6:
    """
    Input is scaled and shifted to that dimensions are in [0, 1]
    Additive with non-overlapping groups, each group has 6 elements (in order)
    """
    has_multiple_minima = False

    def __init__(self, ndim: int):
        assert(ndim % 6 == 0)
        self.ndim = ndim
        self.n_repeat = ndim // 6
        self.alpha = torch.tensor([[1.0, 1.2, 3.0, 3.2]])
        self.A = torch.tensor([[10.0, 3.00, 17.0, 3.50, 1.70, 8.00],
                               [0.05, 10.0, 17.0, 0.10, 8.00, 14.0],
                               [3.00, 3.50, 1.70, 10.0, 17.0, 8.00],
                               [17.0, 8.00, 0.05, 10.0, 0.10, 14.0]]).t()
        self.P = torch.tensor([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                               [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                               [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                               [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]).t()
        min_loc = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]) * 2 - 1
        self.min_loc = min_loc.repeat(self.n_repeat)
        self.min_val = self.eval(self.min_loc).item()
        self.partition = [torch.arange(6) + i*6 for i in range(self.n_repeat)]

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.ndim)
        n = x.size(0)
        x = (x + 1) * 0.5
        output = torch.tensor(0.)
        for i in range(self.n_repeat):
            x_block = x[:, 6 * i:6 * (i + 1)]
            output += -(self.alpha.repeat(n, 1) * torch.exp(-(self.A.unsqueeze(0).repeat(n, 1, 1) * (
                        x_block.unsqueeze(2).repeat(1, 1, 4) -
                        self.P.unsqueeze(0).repeat(n, 1, 1)) ** 2).sum(1))).sum(1, keepdim=True)
        output /= self.n_repeat
        return output


class Rosenbrock:
    """
    Input is scaled and shifted to that dimensions are in [-5, 10]
    Additive with overlapping groups, each group has 2 elements (in order),
    e.g. [0, 1], [1, 2], ...
    """
    has_multiple_minima = False

    def __init__(self, ndim: int):
        self.ndim = ndim
        self.min_loc = (torch.ones(self.ndim) - 2.5) / 7.5
        self.normalizer = 50000.0 / ((90 ** 2 + 9 ** 2) * (self.ndim - 1))
        self.min_val = self.eval(self.min_loc).item()
        self.partition = [torch.arange(2) + i for i in range(self.ndim)]  # additive with overlapping groups

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.ndim)
        x = x * 7.5 + 2.5
        return (100.0 * ((x[:, 1:] - x[:, :-1] ** 2) ** 2) + (x[:, :-1] - 1) ** 2).sum(1,
                                                                                       keepdim=True) * self.normalizer


class Levy:
    """
    Input is scaled and shifted to that dimensions are in [-10, 10]
    """
    has_multiple_minima = False

    def __init__(self, ndim: int):
        self.ndim = ndim
        self.min_loc = torch.ones(self.ndim) / 10.0
        self.min_val = self.eval(self.min_loc).item()
        self.partition = list(torch.arange(self.ndim))  # fully additive

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(1, self.ndim)
        x = x * 10.0
        w = (x - 1.0) / 4.0 + 1.0
        output = torch.sin(math.pi * w[:, :1]) ** 2.0
        for i in range(self.ndim - 1):
            output += (w[:, i:i + 1] - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * w[:, i:i + 1] + 1.0) ** 2.0)
        output += ((w[:, -1:] - 1.0) ** 2 * (1.0 + torch.sin(2 * math.pi * w[:, -1:]) ** 2.0))
        return output
