from typing import Callable, Dict, Optional, Union
from numbers import Number

import torch
from torch.distributions import Distribution, Transform
from gpytorch.distributions import MultivariateNormal

from ..acquisitions import AcquisitionFunction
from ..utils import hessian


def map_estimate(log_prob:       Callable[[torch.Tensor], torch.Tensor],
                 search_space:   Optional[Dict[str, Union[str, Union[str, Number, torch.Tensor, Transform]]]],
                 initial_params: Optional[torch.Tensor] = None,
                 seed:           Optional[int] = 117, **kwargs) -> torch.Tensor:
    if 'dim' not in search_space:
        if initial_params is not None:
            search_space['dim'] = initial_params.numel()
        else:
            raise RuntimeError("Must provide dim or initial_params.")
    acquisition = AcquisitionFunction(lambda p: -log_prob(p), search_space=search_space, seed=seed, **kwargs)
    map_x, aa = acquisition.acquire(guess_point=initial_params)
    return map_x


def laplace(log_prob:            Callable[[torch.Tensor], torch.Tensor],
            search_space:        Optional[Dict[str, Union[str, Union[str, Number, torch.Tensor, Transform]]]] = None,
            initial_params:      Optional[torch.Tensor] = None,
            skip_map_estimation: bool = False,
            seed:                Optional[int] = 117, **kwargs) -> Distribution:
    if skip_map_estimation:
        assert(isinstance(initial_params, torch.Tensor))
        x = initial_params
    else:
        assert(search_space is not None)
        x = map_estimate(log_prob=log_prob, search_space=search_space,
                         initial_params=initial_params, seed=seed, **kwargs)
    x = x.detach().clone()
    h = hessian(lambda p: -log_prob(p), x)
    covar = torch.cholesky_inverse(torch.cholesky(h, upper=True), upper=True)
    return MultivariateNormal(x.detach().clone(), covar)
