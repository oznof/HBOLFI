import gpytorch

from .scale_kernel import make_scaled, ScaleRBFKernel, ScaleMaternKernel
from .cylindrical import CylindricalKernel, original_prediction_strategy_cylindrical
from .patches import kernel_set_priors, additive_kernel_initialize, additive_kernel_set_priors, additive_active_dims
from .patches import fully_additive_set_priors, kernel_out_of_bounds

__all__ = [
    "make_scaled"
    "ScaleRBFKernel",
    "ScaleMaternKernel",
    "CylindricalKernel",
    "original_prediction_strategy_cylindrical"
]

# apply monkey patches related to kernels
gpytorch.kernels.Kernel.set_priors = kernel_set_priors
gpytorch.kernels.Kernel.out_of_bounds = kernel_out_of_bounds
gpytorch.kernels.AdditiveKernel.initialize = additive_kernel_initialize
gpytorch.kernels.AdditiveKernel.set_priors = additive_kernel_set_priors
gpytorch.kernels.AdditiveKernel.active_dims = property(additive_active_dims)
gpytorch.kernels.AdditiveStructureKernel.set_priors = fully_additive_set_priors
