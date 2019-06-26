from .acquisitions import AcquisitionFunction, AdditiveAcquisitionFunction, LCB, LCBAdd
from .bayes_opt import BayesianOptimization, BOLFI
from .distributions import TruncatedNormal, Mixture, make_prior, NormalPrior, MultivariateNormalPrior
from .distributions import GammaPrior, InvGammaPrior, HalfStudentTPrior, LogNormalPrior, DirichletPrior
from .distributions import normal_prob, normal_log_prob, normal_cdf, inverse_normal_cdf, invgamma_solver
from .distributions import biject_to, transform_to, SoftPlusTransform
from .inference import map_estimate, laplace, nuts, GaussianVI
from .kernel_discovery import MHStructureSelector, GibbsStructureSelector, generate_kernel_list
from .kernels import ScaleRBFKernel, ScaleMaternKernel, CylindricalKernel
from .means import QuadPolyMean
from .models import SurrogateModel, AdditiveStructureGPModel, ExactGPModel
from simulators import QuadFormModel, SemiparametricQuadFormModel
from .utils import rng, hessian, plot_contour, partition_metrics, summary_stats


# apply monkey patches
import gpytorch.constraints
import gpytorch.models.exact_prediction_strategies
from .utils import sample_from_prior, register_constraint
from .utils import exact_predictive_covar, interval_transform, interval_inverse_transform, _extract_named_priors
gpytorch.module.Module.sample_from_prior = sample_from_prior
gpytorch.module._extract_named_priors = _extract_named_priors
gpytorch.module.Module.register_constraint = register_constraint
gpytorch.models.exact_prediction_strategies.DefaultPredictionStrategy.exact_predictive_covar = exact_predictive_covar
gpytorch.constraints.Interval.transform = interval_transform
gpytorch.constraints.Interval.inverse_transform = interval_inverse_transform


__all__ = [
    "AcquisitionFunction",
    "AdditiveAcquisitionFunction",
    "LCB",
    "LCBAdd",
    "BayesianOptimization",
    "BOLFI",
    "TruncatedNormal",
    "Mixture",
    "make_prior",
    "biject_to",
    "transform_to",
    "NormalPrior",
    "MultivariateNormalPrior",
    "GammaPrior",
    "InvGammaPrior",
    "HalfStudentTPrior",
    "LogNormalPrior",
    "DirichletPrior",
    "normal_prob",
    "normal_log_prob",
    "normal_cdf",
    "inverse_normal_cdf",
    "invgamma_solver",
    "SoftPlusTransform",
    "map_estimate",
    "laplace",
    "nuts",
    "GaussianVI",
    "MHStructureSelector",
    "GibbsStructureSelector",
    "generate_kernel_list",
    "ScaleRBFKernel",
    "ScaleMaternKernel",
    "CylindricalKernel",
    "QuadPolyMean",
    "SurrogateModel",
    "ExactGPModel",
    "AdditiveStructureGPModel",
    "QuadFormModel",
    "SemiparametricQuadFormModel",
    "rng",
    "hessian",
    "plot_contour",
    "partition_metrics",
    "summary_stats"
]
