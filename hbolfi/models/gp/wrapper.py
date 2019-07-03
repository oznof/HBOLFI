from gpytorch.kernels import Kernel, AdditiveKernel

from .exact_gp import ExactGPModel
from .additive_gp import AdditiveStructureGPModel
from .bayesian_gp import BayesianGPModel


def make_gp(kernel: Kernel, *args, bayesian: bool = False, **kwargs):
    """
    Creates a GP instance depending on `kernel` and `bayesian`.
    :param kernel: `Kernel` instance
    :param bayesian: Indicates whether to create a `BayesianGPModel` instance
    :return: GP instance
    """
    if bayesian:
        return BayesianGPModel(kernel, *args, **kwargs)
    else:
        gp_class = AdditiveStructureGPModel if isinstance(kernel, AdditiveKernel) else ExactGPModel
        return gp_class(kernel, *args, **kwargs)
