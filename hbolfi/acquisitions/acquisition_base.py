from typing import Any, Callable, Dict, Optional, Tuple, Union
from numbers import Number
from time import time

import numpy as np
import scipy
import scipy.optimize

import torch
from torch.nn import Module
from torch.distributions import constraints, biject_to, Transform
from torch.distributions.transforms import AffineTransform, ComposeTransform, SigmoidTransform
from torch.quasirandom import SobolEngine

from .. import settings
from ..models import SurrogateModel
from ..distributions import TruncatedNormal
from ..utils import rng


class AcquisitionFunction(Module):
    def __init__(self,
                 model:             Union[Callable[[torch.Tensor], torch.Tensor], SurrogateModel],
                 search_space:      Optional[Dict[str, Union[str, Union[str, Number, torch.Tensor, Transform]]]] = None,
                 n_inits:           Optional[int] = None,
                 maxiter:           Optional[int] = None,
                 n_candidates:      Optional[int] = None,
                 noise:             Optional[Union[float, torch.Tensor]] = None,
                 # if seed is None, use the current rng state
                 seed:              Optional[int] = None,
                 optimizer:         Optional[str] = None,
                 optimizer_options: Optional[Dict[str, Any]] = None,
                 use_constrained:   bool = False,
                 **kwargs):
        super().__init__()
        self.model = model
        self.search_space = search_space
        self.use_constrained = use_constrained
        if self.use_constrained:
            assert('bounds' in search_space)
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
        self.n_inits = settings.default_n_inits.value() if n_inits is None else n_inits
        self.maxiter = settings.default_maxiter.value() if maxiter is None else maxiter
        self.n_candidates = settings.default_n_candidates.value() if n_candidates is None else n_candidates
        noise = settings.default_acquisition_noise.value() if noise is None else noise
        self.noise = torch.as_tensor(noise)
        self.seed = seed
        self.rng_state = (torch.get_rng_state() if self.seed is None else
                          torch.Generator().manual_seed(self.seed).get_state())
        self.optimizer = settings.default_optimizer.value() if optimizer is None else optimizer
        if self.use_constrained:
            assert(self.optimizer == 'LBFGS')
        if not optimizer_options:
            if self.optimizer in settings.torch_optimizers:
                optimizer_options = settings.default_torch_optimizer_options[self.optimizer]
            elif self.optimizer == 'LBFGS':  # uses scipy
                optimizer_options = {'maxiter': self.maxiter}
            else:
                raise RuntimeError("Invalid optimizer for AcquisitionFunction.")
        self.optimizer_options = optimizer_options
        # other options related to random restarts
        self.fix_sobol = kwargs.pop('fix_sobol', settings.default_fix_sobol.on())
        self.perturb_optimum = kwargs.pop('perturb_optimum', settings.default_perturb_optim.on())
        self.include_inputs = kwargs.pop('include_inputs', settings.default_include_inputs.on())
        self.t = torch.zeros(1)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def optimize(self, x0: torch.Tensor, retain_graph: bool = False) -> Tuple[torch.Tensor, float]:
        if self.dim is None:
            self.dim = x0.numel()
        if self.optimizer == 'LBFGS':
            return self.optimize_lbfgs(x0, retain_graph=retain_graph)

        raw_x = self.transforms['from_real'].inv(x0.view(1, -1)).detach().clone().requires_grad_(True)
        optimizer = settings.torch_optimizers[self.optimizer]([raw_x], **self.optimizer_options)

        def closure() -> float:
            optimizer.zero_grad()
            x = self.transforms['from_real'](raw_x) if self.transforms else raw_x
            loss = self.evaluate(x)
            loss.backward(retain_graph=retain_graph)
            return loss.item()

        prev_loss = None
        for _ in range(self.maxiter):
            curr_loss = optimizer.step(closure)
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

    def optimize_lbfgs(self, x0: torch.Tensor, retain_graph: bool = False) -> Tuple[torch.Tensor, float]:

        def func_and_grad(raw: np.ndarray) -> Tuple[float, np.ndarray]:
            raw = torch.tensor(raw).clone().unsqueeze(0).requires_grad_(True)
            x = raw if self.use_constrained else self.transforms['from_real'](raw)
            loss = self.evaluate(x)
            curr_loss = loss.item()
            loss.backward(retain_graph=retain_graph)
            # if raw.grad.data.norm() > 1:
            #     raw.grad.data.div_(raw.grad.data.norm() + 1e-6)
            # print("curr_loss: ", curr_loss)
            # print(f"{raw.grad.data.norm():.6f}")
            # print(f"loss: {curr_loss}\nx: {x}\ngrad: {raw_x.grad.numpy().squeeze()}")
            return curr_loss, raw.grad.data.clone().double().numpy().squeeze()

        raw_x0 = self.transforms['from_real'].inv(x0)
        raw_x0 = raw_x0.clone().numpy().squeeze()
        bounds = self.search_space['bounds'].t().numpy() if self.use_constrained else None
        res = scipy.optimize.minimize(func_and_grad, raw_x0, method='L-BFGS-B', jac=True,
                                      options=self.optimizer_options, bounds=bounds)
        raw_x = torch.tensor(res['x']).unsqueeze(0)
        # print("raw_x_result: ", raw_x)
        loc = self.transforms['from_real'](raw_x)
        val = self.evaluate(loc).item()
        return loc, val

    def generate_candidates(self, guess_point: Optional[torch.Tensor] = None,
                            verbose: bool = False) -> torch.Tensor:
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
                    n_skip = 0 if self.fix_sobol else torch.randint(self.n_candidates, (1,)).item()
                    x = SobolEngine(self.dim).draw(self.n_candidates + n_skip).to(torch.get_default_dtype())[n_skip:]
                    generate_sobol = False
                elif c == 0 or iters < 10 or torch.rand(1) < 0.5:
                    # keep generating random unif until a valid starting point is found or number of iterations < 10
                    # otherwise, choose randomly (with equal prob) between this option and the next
                    x = torch.rand(self.n_candidates, self.dim)
                else:
                    # apply a random perturbation to the collection of valid starting points
                    x = self.transforms['from_cube'].inv(x0[:c]) + torch.randn(c, self.dim) * 0.01
                    x = torch.min(torch.max(x, torch.zeros(1)), torch.ones(1))  # clip wrt bounds of unit cube
                x = self.transforms['from_cube'](x)
                if c <= 1 and hasattr(self.model, 'train_inputs'):
                    if self.perturb_optimum and 'bounds' in self.search_space:
                        x_opt = self.model.train_inputs[0][torch.argmin(self.model.train_targets)]
                        lb, ub = self.search_space['bounds']
                        x_spray = x_opt + torch.randn(10, x_opt.numel()) * 0.001 * (ub - lb)
                        out_bounds = x_spray < lb
                        x_spray[out_bounds] = 2 * lb.expand(10, self.dim)[out_bounds] - x_spray[out_bounds]
                        out_bounds = x_spray > ub
                        x_spray[out_bounds] = 2 * ub.expand(10, self.dim)[out_bounds] - x_spray[out_bounds]
                        x = torch.cat([x, x_spray], -2)
                    if self.include_inputs:
                        x = torch.cat([x, self.model.train_inputs[0]], -2)
                x_size = x.size(0)
                i = 0
                val = []
                chunk_size = settings.evaluate_chunk_size.value()
                if chunk_size is None:
                    chunk_size = x_size
                start = time()
                if verbose:
                    print(f"Evaluating candidates (iter={iters})...", end='')
                while i < x_size:
                    x_chunk = x[i:i+chunk_size]
                    val.append(self.evaluate(x_chunk))
                    i += x_chunk.size(0)
                val = torch.cat(val)
                ids = torch.argsort(val)[:self.n_inits - c]
                val = val[ids]
                mask_keep = (val == val) & (val != float('inf'))  # remove nan and inf
                ids = ids[mask_keep]
                len_ids = ids.numel()
                if len_ids > 0:
                    x0[c:c+len_ids] = x[ids]
                    c += len_ids
                if verbose:
                    print(f"Done. Gathered: {c}/{self.n_inits}. time={time() - start:.3f}")
                iters += 1
        self.rng_state = rs.generator_or_state
        return x0

    def acquire(self, n: int = 1, guess_point: Optional[torch.Tensor] = None,
                retain_graph: bool = False, verbose: bool = False) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
        self.t += 1
        x0 = self.generate_candidates(guess_point, verbose=verbose)
        # print("guess_point: ",  guess_point)
        # print("candidates (orig): ", x0)
        loc = torch.zeros_like(x0)
        val = torch.zeros(x0.size(0))
        start = time()
        for i, xi in enumerate(x0):
            loc[i], val[i] = self.optimize(xi, retain_graph=retain_graph)
        if verbose:
            print(f"Time in optimize: {time() - start}")
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
        with rng(self.rng_state) as rs:
            x = TruncatedNormal(loc=x, scale=self.noise.sqrt(), low=0., high=1.).sample()
        self.rng_state = rs.generator_or_state
        x = self.transforms['from_cube'](x)
        return x
