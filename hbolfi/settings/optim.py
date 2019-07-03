from gpytorch.settings import _feature_flag, _value_context
from torch.optim import Adam, SGD

torch_optimizers = {'Adam': Adam, 'SGD': SGD}
default_torch_optimizer_options = {'Adam': {'lr': 0.01, 'amsgrad': False}, 'SGD': {'lr': 0.01, 'momentum': 0.0}}


class initialize_before_optimization(_feature_flag):
    _state = True


class initialize_by_sampling(_feature_flag):
    _state = True


class default_optimizer(_value_context):
    _global_value = 'LBFGS'


# default number of optimization restarts
class default_n_inits(_value_context):
    _global_value = 10


# default maximum number of optimization iterations (per run)
class default_maxiter(_value_context):
    _global_value = 1000

# =================================================================================================================
# acquisition-only default values


# noise level to be added to the minimizer (see _add_noise)
class default_acquisition_noise(_value_context):
    _global_value = 0.


# number of points to draw each time a generator is called (see generate_candidates)
class default_n_candidates(_value_context):
    _global_value = 20000
