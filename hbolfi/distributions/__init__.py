from .truncated_normal import TruncatedNormal
from .mixture import Mixture
from .priors import make_prior, NormalPrior, MultivariateNormalPrior, DeltaPrior
from .priors import GammaPrior, InvGammaPrior, HalfStudentTPrior, LogNormalPrior, DirichletPrior
from .priors import SpikeUniformSlabPrior, SpikeNormalSlabPrior
from .utils import normal_prob, normal_log_prob, normal_cdf, inverse_normal_cdf, invgamma_solver, SoftPlusTransform

# update biject_to and transform_to registries, replacing ExpTransform with SoftPlusTransform
from torch.distributions import biject_to, transform_to, constraints, transforms
@biject_to.register(constraints.positive)
@transform_to.register(constraints.positive)
def _transform_to_positive(constraint):
    return SoftPlusTransform()


@biject_to.register(constraints.greater_than)
@biject_to.register(constraints.greater_than_eq)
@transform_to.register(constraints.greater_than)
@transform_to.register(constraints.greater_than_eq)
def _transform_to_greater_than(constraint):
    return transforms.ComposeTransform([SoftPlusTransform(),
                                        transforms.AffineTransform(constraint.lower_bound, 1)])


@biject_to.register(constraints.less_than)
@transform_to.register(constraints.less_than)
def _transform_to_less_than(constraint):
    return transforms.ComposeTransform([SoftPlusTransform(),
                                        transforms.AffineTransform(constraint.upper_bound, -1)])

__all__ = [
    "biject_to",
    "transform_to",
    "TruncatedNormal",
    "Mixture",
    "make_prior",
    "NormalPrior",
    "MultivariateNormalPrior",
    "DeltaPrior",
    "GammaPrior",
    "InvGammaPrior",
    "HalfStudentTPrior",
    "LogNormalPrior",
    "DirichletPrior",
    "SpikeUniformSlabPrior",
    "SpikeNormalSlabPrior",
    "normal_prob",
    "normal_log_prob",
    "normal_cdf",
    "inverse_normal_cdf",
    "invgamma_solver",
    "SoftPlusTransform"
]