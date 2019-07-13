import math
from typing import Optional, Union

import torch
from gpytorch.settings import _feature_flag
from gpytorch.priors import Prior

from torch.distributions import Uniform, constraints, transforms, TransformedDistribution


from ..distributions import InvGammaPrior, StudentTPrior, HalfStudentTPrior, DirichletPrior, SpikeUniformSlabPrior
from ..distributions import make_prior, DeltaPrior, NormalPrior, LogNormalPrior, HalfNormalPrior, ShiftedHalfNormal
from ..distributions import LogStudentTPrior, TruncatedNormal, SpikeNormalSlabPrior, ShiftedInvGamma, invgamma_solver


default_prior_names = ('noise', 'lengthscale', 'outputscale', 'angular_weights', 'alpha', 'beta',
                       'quadpoly_a', 'quadpoly_b', 'quadpoly_c')


class _prior_context(object):
    _prior_registry = {}
    _prior = None
    initial_value = None

    @classmethod
    def default_prior_registry(cls, **kwargs):
        default_priors = {'Delta': DeltaPrior(cls.initial_value, initial_value=cls.initial_value)}
        cls._prior_registry.update(default_priors)
        cls._prior = cls._prior_registry['Delta']

    @classmethod
    def prior(cls, name: Optional[str] = None, reload_default: bool = False, **kwargs) -> Prior:
        if not cls._prior_registry or reload_default:
            cls.default_prior_registry(**kwargs)
        return cls._prior if name is None else cls._prior_registry[name]

    @classmethod
    def switch_default_prior(cls, name: str):
        if not cls._prior_registry:
            cls.default_prior_registry()
        cls._prior = cls._prior_registry[name]

    @classmethod
    def set(cls, prior: Optional[Union[str, Prior]] = None, initial_value: Optional[torch.Tensor] = None,
            name: Optional[str] = None, **kwargs):
        reload_default = False
        if initial_value is not None:
            cls.initial_value = initial_value
            reload_default = True
        if isinstance(prior, str):
            prior = cls.prior(prior, reload_default=reload_default, **kwargs)
        elif isinstance(prior, Prior):
            if not cls._prior_registry:
                cls.default_prior_registry(**kwargs)
            prior.constraint.initial_value = getattr(prior.constraint, 'initial_value', cls.initial_value)
            if name:
                cls._prior_registry[name] = prior
        else:
            raise RuntimeError("Invalid prior.")
        if isinstance(prior, DeltaPrior):
            prior.v = prior.constraint.initial_value
        cls._prior = prior

    @classmethod
    def get(cls):
        return cls._prior, cls.initial_value

    def __init__(self, prior: Optional[Union[str, Prior]] = None, initial_value: Optional[torch.Tensor] = None,
                 name: Optional[str] = None, **kwargs):
        self._orig = self.__class__.get()
        self._instance = (prior, initial_value, name, kwargs)

    def __enter__(self):
        self.__class__.set(*self._instance[:3], **self._instance[-1])

    def __exit__(self, *args):
        self.__class__.set(*self._orig)
        return False


class fix_all_hyperparameters(_feature_flag):
    _state = False


class out_of_bounds_check(_feature_flag):
    _state = False


class noise(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            # to be used with cylindrical kernel (raw parameter in the log space)
            logcauchy = LogStudentTPrior(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                         df=torch.tensor(1.), initial_value=cls.initial_value)
            logcauchy.constraint._support = constraints._GreaterThan(0.)
            logcauchy.constraint._transform = transforms.ExpTransform()
            logcauchy.constraint._inv_transform = logcauchy.constraint._transform.inv

            default_priors = {'Uniform': make_prior(Uniform(torch.tensor(1e-4), torch.tensor(1e10)),
                                                    initial_value=cls.initial_value),
                              'ShiftedHalfNormal': make_prior(ShiftedHalfNormal(torch.tensor(1e-5),
                                                                                torch.tensor(1.)),
                                                              initial_value=cls.initial_value),
                              'HalfNormal': HalfNormalPrior(scale=torch.tensor(1.), initial_value=cls.initial_value),
                              'HalfStudentT': HalfStudentTPrior(scale=torch.tensor(1.), df=torch.tensor(3.),
                                                                initial_value=cls.initial_value),
                              'LogNormal': LogNormalPrior(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                                          initial_value=cls.initial_value),
                              'LogCauchy': logcauchy
                              }
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['HalfStudentT']


# this was found using invgamma_solver, method='normal', l=1/100, u=2
# suitable for search_space [-1, 1]^n
# class invgamma_parameters(_value_context):
#    _global_value = (5.045301507537683, 4.065528015075368)


class lengthscale(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls, lb=1e-4, ub=2.):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            invgamma_parameters = invgamma_solver(lb, ub, method='minimize')
            default_priors = {'InvGamma': InvGammaPrior(*torch.as_tensor(invgamma_parameters),
                                                        initial_value=cls.initial_value),
                              'Uniform':  make_prior(Uniform(torch.as_tensor(lb), torch.as_tensor(ub)),
                                                     initial_value=cls.initial_value),
                              'ShiftedInvGamma': make_prior(ShiftedInvGamma(torch.as_tensor(lb),
                                                                            *torch.as_tensor(invgamma_parameters)),
                                                            initial_value=cls.initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['InvGamma']


# it's less confusing if we handle this type separately
class cylindrical_lengthscale(lengthscale):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls, lb=math.exp(-12), ub=2.):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            # there is a mismatch between the support of the prior and the constraint, but it is required so
            # that we recover the original implementation, where all parameters are in the log domain
            # we are directly setting the transform because the constraint registry biject_to has been
            # changed to use the softplus transform instead of the exp transform
            unif = make_prior(Uniform(torch.as_tensor(lb), torch.as_tensor(ub)),
                              initial_value=cls.initial_value)
            unif.constraint._support = constraints._GreaterThan(0.)
            unif.constraint._transform = transforms.ExpTransform()
            unif.constraint._inv_transform = unif.constraint._transform.inv
            default_priors = {'Uniform': unif}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['Uniform']


class outputscale(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'ShiftedHalfNormal': make_prior(ShiftedHalfNormal(torch.tensor(1e-4), torch.tensor(1.)),
                                                              initial_value=cls.initial_value),
                              'HalfStudentT': HalfStudentTPrior(scale=torch.tensor(1.), df=torch.tensor(3.),
                                                                initial_value=cls.initial_value),
                              'Uniform': make_prior(Uniform(torch.tensor(1e-4), torch.tensor(1e12)),
                                                    initial_value=cls.initial_value),
                              'TruncatedNormal': make_prior(TruncatedNormal(0., 1., 1e-4, 1e6),
                                                            initial_value=cls.initial_value),
                              'HalfNormal': HalfNormalPrior(scale=torch.tensor(1.), initial_value=cls.initial_value),
                              'HalfCauchy': HalfStudentTPrior(scale=torch.tensor(1.), df=torch.tensor(1.),
                                                              initial_value=cls.initial_value),
                              }
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['HalfStudentT']


class cylindrical_outputscale(outputscale):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            # there is a mismatch between the support of the prior and the constraint, but it is required so
            # that we recover the original implementation, where all parameters are in the log domain
            # we are directly setting the transform because the constraint registry biject_to has been
            # changed to use the softplus transform instead of the exp transform
            unif = make_prior(Uniform(torch.tensor(math.exp(-12)), torch.tensor(math.exp(20))),
                              initial_value=cls.initial_value)
            unif.constraint._support = constraints._GreaterThan(0.)
            unif.constraint._transform = transforms.ExpTransform()
            unif.constraint._inv_transform = unif.constraint._transform.inv
            default_priors = {'Uniform': unif}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['Uniform']


class angular_weights(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.ones(4) / 4

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            # the prior in the original implementation
            lognorm = LogNormalPrior(loc=0., scale=2., initial_value=torch.tensor(1.))
            lognorm.constraint._support = constraints._GreaterThan(0.)
            lognorm.constraint._transform = transforms.ExpTransform()
            lognorm.constraint._inv_transform = lognorm.constraint._transform.inv
            default_priors = {'LogNormal': lognorm,
                              'Dirichlet': DirichletPrior(torch.ones(4), initial_value=cls.initial_value),
                              'HalfStudentT': HalfStudentTPrior(scale=torch.tensor(1.), df=torch.tensor(3.),
                                                                initial_value=torch.tensor(1.))}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['LogNormal']


class alpha(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            spikeslab = SpikeUniformSlabPrior(mean=torch.tensor(1.), low=torch.tensor(0.5) - 1e-6,
                                              high=torch.tensor(1.) + 1e-6, spike_scale=torch.tensor(0.01),
                                              initial_value=cls.initial_value)
            spikeslab.constraint._support = constraints._GreaterThan(0.)
            spikeslab.constraint._transform = transforms.ExpTransform()
            spikeslab.constraint._inv_transform = spikeslab.constraint._transform.inv
            default_priors = {'SpikeUniformSlab': spikeslab,
                              'Uniform': make_prior(Uniform(torch.tensor(0.5) - 1e-6, torch.tensor(1.) + 1e6),
                                                    initial_value=cls.initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['SpikeUniformSlab']


class beta(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            spikeslab = SpikeUniformSlabPrior(mean=torch.tensor(1.), low=torch.tensor(1.) - 1e-6,
                                              high=torch.tensor(2.) + 1e-6, spike_scale=torch.tensor(0.01),
                                              initial_value=cls.initial_value)
            spikeslab.constraint._support = constraints._GreaterThan(0.)
            spikeslab.constraint._transform = transforms.ExpTransform()
            spikeslab.constraint._inv_transform = spikeslab.constraint._transform.inv
            default_priors = {'SpikeUniformSlab': spikeslab,
                              'Uniform': make_prior(Uniform(torch.tensor(1.) - 1e-6, torch.tensor(2.) + 1e-6),
                                                    initial_value=cls.initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['SpikeUniformSlab']


class quadpoly_a(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'HalfNormal': HalfNormalPrior(scale=torch.tensor(1.), initial_value=cls.initial_value),
                              'HalfStudentT': HalfStudentTPrior(scale=torch.tensor(1.), df=torch.tensor(3.),
                                                                initial_value=cls.initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['HalfNormal']


class quadpoly_b(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(0.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'Normal': NormalPrior(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                                    initial_value=cls.initial_value),
                              'StudentT': StudentTPrior(scale=torch.tensor(1.), df=torch.tensor(3.),
                                                        initial_value=cls.initial_value),
                              'SpikeNormalSlab': SpikeNormalSlabPrior(mean=torch.tensor(0.),
                                                                      low=torch.tensor(-1e5),
                                                                      high=torch.tensor(1e5),
                                                                      spike_scale=torch.tensor(1e-6),
                                                                      slab_scale=torch.tensor(1.),
                                                                      initial_value=cls.initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['Normal']


class quadpoly_c(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(0.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'Normal': NormalPrior(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                                    initial_value=cls.initial_value),
                              'StudentT': StudentTPrior(scale=torch.tensor(1.), df=torch.tensor(3.),
                                                        initial_value=cls.initial_value),
                              'SpikeNormalSlab': SpikeNormalSlabPrior(mean=torch.tensor(0.),
                                                                      low=torch.tensor(-1e5),
                                                                      high=torch.tensor(1e5),
                                                                      spike_scale=torch.tensor(1e-6),
                                                                      slab_scale=torch.tensor(1.),
                                                                      initial_value=cls.initial_value),
                              # this distribution is to be used with dynamically_adjust_quadpoly_c,
                              # so that the values are adjusted to the observed data
                              'Uniform': make_prior(Uniform(low=torch.tensor(-10.), high=torch.tensor(10.)),
                                                    initial_value=cls.initial_value)
                              }
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['Normal']


# only applies if the default is Normal or Uniform
class dynamically_adjust_quadpoly_c(_feature_flag):
    _state = True
