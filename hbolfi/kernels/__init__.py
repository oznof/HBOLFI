import gpytorch

from .scale_kernel import make_scaled, ScaleRBFKernel, ScaleMaternKernel
from .cylindrical import CylindricalKernel
from .patches import kernel_set_priors, additive_kernel_initialize, additive_kernel_set_priors


__all__ = [
    "make_scaled"
    "ScaleRBFKernel",
    "ScaleMaternKernel",
    "CylindricalKernel"
]

# apply monkey patches related to kernels
gpytorch.kernels.Kernel.set_priors = kernel_set_priors
gpytorch.kernels.AdditiveKernel.initialize = additive_kernel_initialize
gpytorch.kernels.AdditiveKernel.set_priors = additive_kernel_set_priors
