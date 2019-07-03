from typing import Callable, Dict, Optional, Union
from numbers import Number

import torch
from torch.distributions import Distribution, Transform
from gpytorch.distributions import MultivariateNormal
from gpytorch import add_jitter

from .. import acquisitions
from ..utils import hessian


def map_estimate(log_prob:       Callable[[torch.Tensor], torch.Tensor],
                 search_space:   Optional[Dict[str, Union[str, Union[str, Number, torch.Tensor, Transform]]]],
                 initial_params: Optional[torch.Tensor] = None,
                 seed:           Optional[int] = None, **kwargs) -> torch.Tensor:
    if 'dim' not in search_space:
        if initial_params is not None:
            search_space['dim'] = initial_params.numel()
        else:
            raise RuntimeError("Must provide dim or initial_params.")
    acq_fn = acquisitions.AcquisitionFunction(lambda p: -log_prob(p), search_space=search_space, seed=seed, **kwargs)
    map_x, aa = acq_fn.acquire(guess_point=initial_params)
    return map_x


def laplace(log_prob:            Callable[[torch.Tensor], torch.Tensor],
            search_space:        Optional[Dict[str, Union[str, Union[str, Number, torch.Tensor, Transform]]]] = None,
            initial_params:      Optional[torch.Tensor] = None,
            skip_map_estimation: bool = False,
            seed:                Optional[int] = None, **kwargs) -> Distribution:
    if skip_map_estimation:
        assert(isinstance(initial_params, torch.Tensor))
        x = initial_params
    else:
        assert(search_space is not None)
        x = map_estimate(log_prob=log_prob, search_space=search_space,
                         initial_params=initial_params, seed=seed, **kwargs)
    x = x.detach().clone()
    h = hessian(lambda p: -log_prob(p), x)
    try:
        covar = torch.cholesky_inverse(torch.cholesky(h, upper=True), upper=True)
    except RuntimeError:
        print(h)
        covar = torch.cholesky_inverse(torch.cholesky(add_jitter(h, 1e-3), upper=True), upper=True)
    return MultivariateNormal(x.detach().clone(), covar)
