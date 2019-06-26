from typing import Any, Callable, Dict, Optional, Tuple, Union
from numbers import Number
from time import time

import numpy as np
import scipy
import scipy.optimize

import torch
from torch.nn import Module
from torch.optim import Adam, SGD
from torch.distributions import constraints, biject_to, Transform
from torch.distributions.transforms import AffineTransform, ComposeTransform, SigmoidTransform
from torch.quasirandom import SobolEngine

from ..models import SurrogateModel
from ..distributions import TruncatedNormal
from ..utils import rng

DEFAULT_SEED = 117
# Optimization methods
OPTIMIZERS = {'Adam': Adam, 'SGD': SGD}  # only need to list pytorch optimizers, LBFGS is from scipy
DEFAULT_OPTIMIZER = 'LBFGS'
DEFAULT_OPTIMIZER_OPTIONS = {'Adam': {'lr': 0.01, 'amsgrad': False}, 'SGD': {'lr': 0.01, 'momentum': 0.0}}
# NOTE: Optimization is done in the unconstrained space. Constrained parameters are transformed.
#       Adam is not recommended. LBFGS runs faster than SGD (due to its implementation) and is more rpbust.
DEFAULT_N_INITS = 10    # default number of optimization restarts
DEFAULT_MAXITER = 1000  # default maximum number of optimization iterations (per run)
NUM_SAMPLES_INIT = 20000  # number of points to draw each time a generator is called (see generate_candidates)


class AcquisitionFunction(Module):
    def __init__(self,
                 model:             Union[Callable[[torch.Tensor], torch.Tensor], SurrogateModel],
                 search_space:      Optional[Dict[str, Union[str, Union[str, Number, torch.Tensor, Transform]]]] = None,
                 n_inits:           int = DEFAULT_N_INITS,
                 maxiter:           int = DEFAULT_MAXITER,
                 noise:             Union[float, torch.Tensor] = 0.,
                 seed:              Optional[int] = DEFAULT_SEED,
                 optimizer:         str = DEFAULT_OPTIMIZER,
                 optimizer_options: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__()
        self.model = model
        self.search_space = search_space
        transforms = {}
        if search_space:
            if 'dim' in search_space:
                self.dim = search_space['dim']
            transforms = {p: v for p, v in search_space.items() if p in ('from_cube', 'from_real')}
            if len(transforms) == 2:
                pass
            elif 'from_real' in transforms:
                transforms['from_cube'] = ComposeTransform([SigmoidTransform().inv, transforms['from_real']])
            elif 'from_cube' in transforms:
                transforms['from_real'] = ComposeTransform([SigmoidTransform(), transforms['from_cube']])
            elif 'bounds' in search_space:
                bounds = search_space['bounds']
                self.dim = bounds[0].numel()
                transforms = {'from_cube': AffineTransform(bounds[0], bounds[1] - bounds[0], 0),
                              'from_real': biject_to(constraints.interval(bounds[0], bounds[1]))}
            else:
                raise RuntimeError("Must specify search_space by 'bounds' or implicitly by the maps"
                                   "'from_cube' or 'from_real'")
        self.transforms = transforms
        self.n_inits = n_inits
        self.maxiter = maxiter
        self.noise = torch.as_tensor(noise)
        self.seed = seed
        self.rng_state = (torch.get_rng_state() if self.seed is None else
                          torch.Generator().manual_seed(self.seed).get_state())
        self.optimizer = optimizer
        if not optimizer_options:
            if optimizer in OPTIMIZERS:  # pytorch optimizers
                optimizer_options = DEFAULT_OPTIMIZER_OPTIONS[optimizer]
            elif optimizer == 'LBFGS':  # uses scipy
                optimizer_options = {'maxiter': self.maxiter}
            else:
                raise RuntimeError("Invalid optimizer for AcquisitionFunction.")
        self.optimizer_options = optimizer_options
        self.t = torch.zeros(1)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def optimize(self, x0: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self.dim is None:
            self.dim = x0.numel()
        if self.optimizer == 'LBFGS':
            return self.optimize_lbfgs(x0)

        raw_x = self.transforms['from_real'].inv(x0.view(1, -1)).detach().clone().requires_grad_(True)
        optimizer = OPTIMIZERS[self.optimizer]([raw_x], **self.optimizer_options)

        def closure() -> float:
            optimizer.zero_grad()
            x = self.transforms['from_real'](raw_x) if self.transforms else raw_x
            loss = self.evaluate(x)
            loss.backward()
            return loss.item()

        prev_loss = None
        for _ in range(self.maxiter):
            curr_loss = optimizer.step(closure).item()
            ftol = abs(prev_loss - curr_loss) / max(1, abs(prev_loss), abs(curr_loss)) if prev_loss else 1
            # gtol = raw_x.grad.abs().sum()
            # print(f"loss:{curr_loss}\nftol: {ftol}\ngrad: {raw_x.grad}\ngtol: {gtol}\n")
            if ftol <= 1e-9:
                break
            prev_loss = curr_loss
        raw_x = raw_x.detach().clone()
        loc = self.transforms['from_real'](raw_x)
        val = self.evaluate(loc).item()
        return loc, val

    def optimize_lbfgs(self, x0: torch.Tensor) -> Tuple[torch.Tensor, float]:

        def func_and_grad(raw: np.ndarray) -> Tuple[float, np.ndarray]:
            raw = torch.tensor(raw).unsqueeze(0).requires_grad_(True)
            x = self.transforms['from_real'](raw) if self.transforms else raw
            loss = self.evaluate(x)
            curr_loss = loss.item()
            loss.backward()
            # print(f"loss: {curr_loss}\nx: {x}\ngrad: {raw_x.grad.numpy().squeeze()}")
            return curr_loss, raw.grad.clone().numpy().squeeze()

        raw_x0 = self.transforms['from_real'].inv(x0)
        raw_x0 = raw_x0.clone().numpy().squeeze()
        res = scipy.optimize.minimize(func_and_grad, raw_x0, method='L-BFGS-B', jac=True,
                                      options=self.optimizer_options,
                                      bounds=None)  # use transformation to perform unconstrained optimization
        raw_x = torch.tensor(res['x']).unsqueeze(0)
        loc = self.transforms['from_real'](raw_x)
        val = self.evaluate(loc).item()
        return loc, val

    def generate_candidates(self, guess_point: Optional[torch.Tensor] = None) -> torch.Tensor:
        with rng(self.rng_state) as rs:
            c = 0
            if self.dim is None:  # must provide guess_point, or it will break here
                self.dim = guess_point.numel()
            x0 = torch.zeros(self.n_inits, self.dim)
            if isinstance(guess_point, torch.Tensor):
                val = self.evaluate(guess_point.view(1, -1))
                if not torch.isnan(val) and val != float('inf'):
                    x0[0] = guess_point
                    c += 1
            iters = 0
            generate_sobol = True
            while c < self.n_inits:
                if generate_sobol:
                    # generate sobol the first time
                    x = SobolEngine(self.dim).draw(NUM_SAMPLES_INIT).to(torch.get_default_dtype())
                    generate_sobol = False
                elif c == 0 or iters < 10 or torch.rand(1) < 0.5:
                    # keep generating random unif until a valid starting point is found or number of iterations < 10
                    # otherwise, choose randomly (with equal prob) between this option and the next
                    x = torch.rand(NUM_SAMPLES_INIT, self.dim)
                else:
                    # apply a random perturbation to the collection of valid starting points
                    x = self.transforms['from_cube'].inv(x0[:c]) + torch.randn(c, self.dim) * 0.01
                    x = torch.min(torch.max(x, torch.zeros(1)), torch.ones(1))  # clip wrt bounds of unit cube
                x = self.transforms['from_cube'](x)
                val = self.evaluate(x)
                ids = torch.argsort(val)[:self.n_inits - c]
                val = val[ids]
                mask_keep = (val == val) & (val != float('inf'))  # remove nan and inf
                ids = ids[mask_keep]
                len_ids = ids.numel()
                if len_ids > 0:
                    x0[c:c+len_ids] = x[ids]
                    c += len_ids
                iters += 1
        self.rng_state = rs.generator_or_state
        return x0

    def acquire(self, n: int = 1, guess_point: Optional[torch.Tensor] = None,
                verbose: bool = False) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
        self.t += 1
        x0 = self.generate_candidates(guess_point)
        loc = torch.zeros_like(x0)
        val = torch.zeros(x0.size(0))
        start = time()
        for i, xi in enumerate(x0):
            loc[i], val[i] = self.optimize(xi)
        if verbose:
            print(f"total time in acquisition: {time() - start}")
        i = torch.argmin(val)
        xbest = loc[i]
        valbest = val[i].item()
        if torch.any(self.noise != 0):
            xbest = self._add_noise(xbest.repeat(n, 1)).squeeze()
            valbest = self.evaluate(xbest)
        return xbest, valbest

    # noise is added on the unit cube
    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transforms['from_cube'].inv(x)
        noise = self.noise.expand((self.dim,))
        with rng(self.rng_state) as rs:
            x = TruncatedNormal(loc=x, scale=noise.sqrt(), low=0., high=1.).sample()
        self.rng_state = rs.generator_or_state
        x = self.transforms['from_cube'](x)
        return x
