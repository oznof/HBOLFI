from typing import Optional, Union

import torch
from gpytorch.settings import _feature_flag
from gpytorch.priors import Prior

from torch.distributions import Uniform


from ..distributions import InvGammaPrior, StudentTPrior, HalfStudentTPrior, DirichletPrior, SpikeUniformSlabPrior
from ..distributions import make_prior, DeltaPrior, NormalPrior, HalfNormalPrior, ShiftedHalfNormal
from ..distributions import TruncatedNormal, SpikeNormalSlabPrior, ShiftedInvGamma, invgamma_solver


default_prior_names = ('noise', 'lengthscale', 'outputscale', 'angular_weights', 'alpha', 'beta',
                       'quadpoly_a', 'quadpoly_b', 'quadpoly_c')


class _prior_context(object):
    _prior_registry = {}
    _prior = None
    initial_value = None

    @classmethod
    def default_prior_registry(cls):
        default_priors = {'Delta': DeltaPrior(cls.initial_value, initial_value=cls.initial_value)}
        cls._prior_registry.update(default_priors)
        cls._prior = cls._prior_registry['Delta']

    @classmethod
    def prior(cls, name: Optional[str] = None, reload_default: bool = False) -> Prior:
        if not cls._prior_registry or reload_default:
            cls.default_prior_registry()
        return cls._prior if name is None else cls._prior_registry[name]

    @classmethod
    def switch_default_prior(cls, name: str):
        if not cls._prior_registry:
            cls.default_prior_registry()
        cls._prior = cls._prior_registry[name]

    @classmethod
    def set(cls, prior: Optional[Union[str, Prior]] = None, initial_value: Optional[torch.Tensor] = None,
            name: Optional[str] = None):
        reload_default = False
        if initial_value is not None:
            cls.initial_value = initial_value
            reload_default = True
        if isinstance(prior, str):
            prior = cls.prior(prior, reload_default=reload_default)
        elif isinstance(prior, Prior):
            if not cls._prior_registry:
                cls.default_prior_registry()
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

    def __init__(self, prior: Optional[Union[str, Prior]] = None, initial_value: Optional[torch.Tensor] = None):
        self._orig = self.__class__.get()
        self._instance = (prior, initial_value)

    def __enter__(self):
        self.__class__.set(*self._instance)

    def __exit__(self, *args):
        self.__class__.set(*self._orig)
        return False


class fix_all_hyperparameters(_feature_flag):
    _state = False


class noise(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'Uniform': make_prior(Uniform(torch.tensor(1e-4), torch.tensor(1e10)),
                                                    initial_value=cls.initial_value),
                              'ShiftedHalfNormal': make_prior(ShiftedHalfNormal(torch.tensor(1e-5),
                                                                                torch.tensor(1.)),
                                                              initial_value=cls.initial_value),
                              'HalfNormal': HalfNormalPrior(scale=torch.tensor(1.), initial_value=cls.initial_value),
                              'HalfStudentT': HalfStudentTPrior(scale=torch.tensor(1.), df=torch.tensor(3.),
                                                                initial_value=cls.initial_value),
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
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            invgamma_parameters = invgamma_solver(1/100, 2., method='minimize')
            default_priors = {'InvGamma': InvGammaPrior(*invgamma_parameters, initial_value=cls.initial_value),
                              'Uniform':  make_prior(Uniform(torch.tensor(1e-4), torch.tensor(2.)),
                                                     initial_value=cls.initial_value),
                              'ShiftedInvGamma': make_prior(ShiftedInvGamma(torch.tensor(1e-4), *invgamma_parameters),
                                                            initial_value=cls.initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['InvGamma']


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
                              'HalfNormal': HalfNormalPrior(scale=torch.tensor(1.), initial_value=cls.initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['HalfStudentT']


class angular_weights(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.ones(4) / 4

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'Dirichlet': DirichletPrior(torch.ones(4), initial_value=cls.initial_value),
                              'HalfNormal': HalfNormalPrior(scale=torch.tensor(1.), initial_value=torch.tensor(1.)),
                              'HalfStudentT': HalfStudentTPrior(scale=torch.tensor(1.), df=torch.tensor(3.),
                                                                initial_value=torch.tensor(1.))}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['HalfNormal']


class alpha(_prior_context):
    _prior_registry = {}
    _prior = None
    initial_value = torch.tensor(1.)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'SpikeUniformSlab': SpikeUniformSlabPrior(mean=torch.tensor(1.), low=torch.tensor(0.5),
                                                                        high=torch.tensor(1.),
                                                                        spike_scale=torch.tensor(0.01),
                                                                        initial_value=cls.initial_value),
                              'Uniform': make_prior(Uniform(torch.tensor(0.5), torch.tensor(1.)),
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
            default_priors = {'SpikeUniformSlab': SpikeUniformSlabPrior(mean=torch.tensor(1.), low=torch.tensor(1.),
                                                                        high=torch.tensor(2.),
                                                                        spike_scale=torch.tensor(0.01),
                                                                        initial_value=cls.initial_value),
                              'Uniform': make_prior(Uniform(torch.tensor(1.), torch.tensor(2.)),
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
                                                                      initial_value=cls.initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['Normal']


class dynamically_adjust_quadpoly_c(_feature_flag):
    _state = True
