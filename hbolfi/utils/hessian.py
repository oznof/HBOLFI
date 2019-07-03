from typing import Callable

import numpy as np
import torch


def hessian(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
            eps: float = 1e-8) -> torch.Tensor:
    x.requires_grad_(True)
    # remove cdist temporarily, so that it is not used when computing distances
    # reason: no implementation of the derivative of cdist backward
    cdist = None
    if hasattr(torch, 'cdist'):
        cdist = torch.cdist
        delattr(torch, 'cdist')
    y = fn(x)
    grads = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
    n = x.size(-1)
    h = torch.zeros((n, n))
    for i in range(n):
        if x.grad is not None:
            x.grad.data.zero_()
        grads[i].backward(retain_graph=(i != n - 1))
        h[:, i] = x.grad
    if cdist:
        setattr(torch, "cdist", cdist)
    h[np.diag_indices(h.shape[0])] += eps
    return h
