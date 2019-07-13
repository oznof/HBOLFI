from .constraint import Constraint
from .truncated_normal import TruncatedNormal
from .mixture import Mixture
from .priors import make_prior, NormalPrior, MultivariateNormalPrior, HalfNormalPrior
from .priors import GammaPrior, InvGammaPrior, StudentTPrior, HalfStudentTPrior, LogStudentTPrior
from .priors import LogNormalPrior, DirichletPrior
from .priors import DeltaPrior, SpikeUniformSlabPrior, SpikeNormalSlabPrior
from .utils import normal_prob, normal_log_prob, normal_cdf, inverse_normal_cdf, invgamma_solver, SoftPlusTransform
from .shifted_half_normal import ShiftedHalfNormal
from .inverse_gamma import ShiftedInvGamma
from .training_distribution import TrainingDistribution

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
    "Constraint",
    "biject_to",
    "transform_to",
    "TruncatedNormal",
    "Mixture",
    "make_prior",
    "NormalPrior",
    "MultivariateNormalPrior",
    "HalfNormalPrior",
    "ShiftedHalfNormal",
    "ShiftedInvGamma",
    "DeltaPrior",
    "GammaPrior",
    "InvGammaPrior",
    "StudentTPrior",
    "LogStudentTPrior",
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
    "SoftPlusTransform",
    "TrainingDistribution"
]
