from gpytorch.settings import _feature_flag, _value_context
from torch.optim import Adam, SGD

torch_optimizers = {'Adam': Adam, 'SGD': SGD}
default_torch_optimizer_options = {'Adam': {'lr': 0.01, 'amsgrad': False}, 'SGD': {'lr': 0.01, 'momentum': 0.0}}


# NOTE: this also applies to BayesianGP.fit
class initialize_before_fitting(_feature_flag):
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


# whether to use a fixed sobol sequence during the generation of candidates
class default_fix_sobol(_feature_flag):
    _state = False


# whether to apply a random perturbation to the current optimum and add these points as candidates
class default_perturb_optim(_feature_flag):
    _state = True


# whether to include all the input data in the candidate set
class default_include_inputs(_feature_flag):
    _state = True


# the set of candidates can be large, and it might be infeasible to evaluate all points in a single call
# it's more efficient to split the set into chunks. To disable splitting, set to None.
# It might be better to set it to a smaller value when using GPU.
class evaluate_chunk_size(_value_context):
    _global_value = 1000
