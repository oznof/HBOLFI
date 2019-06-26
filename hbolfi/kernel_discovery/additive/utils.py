from typing import Optional

import torch
from gpytorch.kernels import Kernel

from ...kernels import ScaleRBFKernel


def generate_kernel_list(num_groups: int, num_dims: int, generator: Optional[torch.Generator] = None,
                         kernel: Kernel = ScaleRBFKernel, ard: bool = False):
    """
    Generate a kernel_list given M groups such that
    alpha = ones(num_groups), concentration parameters
    eta ~ Dir(alpha), mixing proportions
    z_d ~ Cat(eta), for d in range(dim), assignments
    """
    alpha = torch.ones(num_groups)
    p_dist = torch._sample_dirichlet(alpha, generator=generator)
    z = torch.multinomial(p_dist, num_dims, replacement=True, generator=generator)
    k_list = []
    for v in torch.unique(z):
        active_dims = torch.nonzero(z == v).squeeze()
        if active_dims.numel():
            k = kernel(active_dims=active_dims, ard_num_dims=(active_dims.numel() if ard else None))
            k_list.append(k)
    return k_list
