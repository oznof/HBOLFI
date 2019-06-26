import torch
from gpytorch.settings import _feature_flag, _value_context

from torch.distributions import Uniform

from .distributions import InvGammaPrior, HalfStudentTPrior, DirichletPrior, SpikeUniformSlabPrior
from .distributions import make_prior, DeltaPrior


class _prior_context(object):
    _prior_registry = {}
    _prior = None
    _initial_value = None

    @classmethod
    def default_prior_registry(cls):
        default_priors = {'Delta': DeltaPrior(cls._initial_value, initial_value=cls._initial_value)}
        cls._prior_registry.update(default_priors)
        cls._prior = cls._prior_registry['Delta']

    @classmethod
    def prior(cls, name=None, reload_default=False):
        if not cls._prior_registry or reload_default:
            cls.default_prior_registry()
        return cls._prior if name is None else cls._prior_registry[name]

    @classmethod
    def initial_value(cls):
        return cls._initial_value

    @classmethod
    def switch_default_prior(cls, name):
        if not cls._prior_registry:
            cls.default_prior_registry()
        cls._prior = cls._prior_registry[name]

    @classmethod
    def set(cls, prior=None, initial_value=None, sample_value=False, force=False):
        prior = cls._prior_registry[prior] if isinstance(prior, str) else prior
        cls._prior = prior
        if prior:
            initial_value = prior.constraint.__dict__.get('initial_value', initial_value)
        if initial_value is not None or force:
            cls._initial_value = initial_value
        cls._sample_value._state = sample_value

    @classmethod
    def get(cls):
        return cls._prior, cls._initial_value, cls._sample_value.on()

    def __init__(self, prior=None, initial_value=None, sample_value=False, replace=False):
        self._orig = self.__class__.get()
        if prior:
            prior = self.prior(prior) if isinstance(prior, str) else prior
            initial_value = prior.constraint.__dict__.get('initial_value', initial_value)
            prior.constraint.initial_value = initial_value
        self._instance = (prior, initial_value, sample_value)
        self._replace = replace

    def __enter__(self):
        self.__class__.set(*self._instance, force=self._replace)

    def __exit__(self, *args):
        if not self._replace:
            self.__class__.set(*self._orig, force=True)
        self._replace = False
        return False


class fix_all_hyperparameters(_feature_flag):
    _state = False

class initialize_before_optimization(_feature_flag):
    _state = True

class noise(_prior_context):
    _prior_registry = {}
    _prior = None
    _initial_value = torch.ones(1)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'Uniform': make_prior(Uniform(torch.Tensor([1e-4]), torch.Tensor([1e4])),
                                                    initial_value=cls._initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['Uniform']



# this was found using invgamma_solver, method='normal', l=1/100, u=2
# suitable for search_space [-1, 1]^n
class invgamma_parameters(_value_context):
    _global_value = (5.045301507537683, 4.065528015075368)


class lengthscale(_prior_context):
    _prior_registry = {}
    _prior = None
    _initial_value = torch.ones(1)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'InvGamma': InvGammaPrior(*invgamma_parameters.value(), initial_value=cls._initial_value),
                              'Uniform':  make_prior(Uniform(torch.Tensor([0.01]), torch.Tensor([2.])),
                                                  initial_value=cls._initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['InvGamma']


class outputscale(_prior_context):
    _prior_registry = {}
    _prior = None
    _initial_value = torch.ones(1)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'HalfStudentT': HalfStudentTPrior(scale=torch.ones(1), df=torch.Tensor([3.]),
                                                                initial_value=cls._initial_value),
                              'Uniform': make_prior(Uniform(torch.Tensor([6e-6]), torch.Tensor([1e5])),
                                                    initial_value=cls._initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['HalfStudentT']


class angular_weights(_prior_context):
    _prior_registry = {}
    _prior = None
    _initial_value = torch.ones(4) / 4

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'Dirichlet': DirichletPrior(torch.ones(4), initial_value=cls._initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['Dirichlet']


class alpha(_prior_context):
    _prior_registry = {}
    _prior = None
    _initial_value = torch.ones(1)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'SpikeUniformSlab': SpikeUniformSlabPrior(mean=torch.ones(1), low=torch.Tensor([0.5]),
                                                                        high=torch.ones(1),
                                                                        spike_scale=torch.Tensor([0.01]),
                                                                        initial_value=cls._initial_value)}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['SpikeUniformSlab']


class beta(_prior_context):
    _prior_registry = {}
    _prior = None
    _initial_value = torch.ones(1)

    @classmethod
    def default_prior_registry(cls):
        super().default_prior_registry()
        if fix_all_hyperparameters.off():
            default_priors = {'SpikeUniformSlab': SpikeUniformSlabPrior(mean=torch.ones(1), low=torch.ones(1),
                                                                        high=torch.Tensor([2.]),
                                                                        spike_scale=torch.Tensor([0.01]),
                                                                        initial_value=torch.ones(1))}
            cls._prior_registry.update(default_priors)
            cls._prior = cls._prior_registry['SpikeUniformSlab']

