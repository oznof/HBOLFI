import math
from typing import List, Dict, Optional
import copy

import sampyl

import torch
import gpytorch
from gpytorch.models.gp import GP
from gpytorch.kernels import Kernel, AdditiveKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..surrogate_model import SurrogateModel
from .exact_gp import ExactGPModel
from .additive_gp import AdditiveStructureGPModel
from ...inference import nuts
from ...distributions import Constraint


class BayesianGPModel(GP, SurrogateModel):
    def __init__(self, kernel: Kernel, *args,
                 sampler: str = 'slice',
                 sampler_options: Optional[dict] = None,
                 **kwargs):
        self.__dict__['_initialized'] = False
        GP.__init__(self)
        SurrogateModel.__init__(self)
        delattr(self, 'train_inputs')
        delattr(self, 'train_targets')
        delattr(self, 'likelihood')
        self.gp_class = AdditiveStructureGPModel if isinstance(kernel, AdditiveKernel) else ExactGPModel
        self.samples = [self.gp_class(kernel, *args, **kwargs)]
        self.sampler = sampler
        if sampler_options is None:
            sampler_options = {'num_samples': 10, 'warmup_steps': 20, 'num_thinning': 1, 'num_chains': 1}
        self.sampler_options = sampler_options
        self._initialized = True

    @property
    def log_likelihood(self):
        if self.samples[0].log_likelihood is None:
            return None
        return torch.tensor([sample.log_likelihood for sample in self.samples]).logsumexp(-1).item() \
            - math.log(len(self.samples))

    def set_train_data(self, inputs=None, targets=None, **kwargs):
        base_sample = self.samples[0]
        base_sample.set_train_data(inputs, targets, **kwargs)
        # make all samples point to the same data objects
        if len(self.samples) > 1:
            for sample in self.samples[1:]:
                sample.train_inputs = base_sample.train_inputs
                sample.train_targets = base_sample.train_targets
                sample.prediction_strategy = None

    def get_fantasy_model(self, inputs, targets, **kwargs):
        self.samples = [sample.get_fantasy_model(inputs, targets, **kwargs) for sample in self.samples]
        if len(self.samples) > 1:
            # make all samples point to the same data objects
            base_sample = self.samples[0]
            for sample in self.samples[1:]:
                sample.train_inputs = base_sample.train_inputs
                sample.train_targets = base_sample.train_targets
        return self

    def __call__(self, x: torch.Tensor) -> MultivariateNormal:
        mvns = [sample(x) for sample in self.samples]
        mean = torch.stack([mvn.mean for mvn in mvns])
        covar = gpytorch.lazify(torch.stack([mvn.covariance_matrix for mvn in mvns]))
        return MultivariateNormal(mean, covar)

    @staticmethod
    def transform_jacobian(model: ExactGPModel, params: Dict[str, torch.Tensor],
                           transforms: Optional[Dict[str, Constraint]] = None) -> torch.Tensor:
        r = 0
        for name, param in params.items():
            constraint = model.constraint_for_parameter_name(name) if transforms is None else transforms.get(name)
            if constraint is not None:
                r = r + constraint._transform.log_abs_det_jacobian(param, constraint._transform(param)).sum()
        return r

    @staticmethod
    def potential_function(params: Dict[str, torch.Tensor], model: ExactGPModel,
                           mll: ExactMarginalLogLikelihood,
                           transforms: Optional[Dict[str, Constraint]] = None,
                           load_params: bool = True) -> torch.Tensor:
        # TODO: Catch exception
        # try:
        if load_params:  # if False, it is assumed params have already been loaded
            model._load_raw_parameters(**params)
        output = model(model.train_inputs[0])
        log_joint = mll(output, model.train_targets) * model.train_targets.numel()  # mll returns log_prob / target_size
        log_joint = log_joint + BayesianGPModel.transform_jacobian(model, params, transforms)
        return -log_joint
        # except RuntimeError:  # return `nan` instead of raise Value Error if not be able to get Cholesky
        #     r = 0
        #     for p in params.values():
        #         r = r + 0 * p.sum()
        #     return torch.tensor(float('nan')) + r

    def fit(self, train_inputs: Optional[torch.Tensor] = None,
            train_targets: Optional[torch.Tensor] = None, verbose=False):
        base_sample = self.samples[0]
        base_sample.load_init()  # NOTE: might want to set the respective context manager to False
        base_sample.train()
        if train_inputs is not None and train_targets is not None:
            base_sample.set_train_data(inputs=train_inputs, targets=train_targets, strict=False)
        mll = ExactMarginalLogLikelihood(base_sample.likelihood, base_sample)
        if self.sampler == 'nuts':
            param_state = base_sample.param_state
            transforms = {name: base_sample.constraint_for_parameter_name(name) for name in param_state}

            def potential_fn(params):
                return BayesianGPModel.potential_function(params, model=base_sample, mll=mll, transforms=transforms)

            param_states = nuts(potential_fn=potential_fn, initial_params=param_state,
                                get_exec_traces=True, **self.sampler_options)
            self.samples = [base_sample.clone()._load_raw_parameters(**state) for state in param_states]
        elif self.sampler == 'slice':
            param_state = base_sample.param_state
            transforms = {name: base_sample.constraint_for_parameter_name(name) for name in param_state}

            def logp(param):
                base_sample.load_array_state(param, param_state)  # modifies the values of param_state
                logp = -BayesianGPModel.potential_function(param_state, model=base_sample, mll=mll,
                                                           transforms=transforms, load_params=False).item()
                if logp != logp:
                    logp = -float('inf')
                return logp
            sampler = sampyl.Slice(logp=logp, start={'param': base_sample.state_to_array(param_state)}, compwise=True)
            warmup_steps = self.sampler_options.get('warmup_steps', 100)
            num_thinning = self.sampler_options.get('num_thinning', 10)
            num_samples = self.sampler_options.get('num_samples', 10)
            param_states = sampler.sample(num=warmup_steps + num_thinning * num_samples,
                                          burn=warmup_steps, thin=num_thinning)
            self.samples = [base_sample.clone().load_array_state(state[0]) for state in param_states]
        else:
            raise NotImplementedError(f"{self.sampler} is not a valid sampler.")
        for sample in self.samples:
            sample.eval()

    def __getattr__(self, item):
        try:
            return GP.__getattr__(self, item)
        except AttributeError:
            return getattr(self.samples[0], item)

    def __setattr__(self, name, value):
        if not self._initialized:
            GP.__setattr__(self, name, value)
        else:
            if name in self.__dict__:
                GP.__setattr__(self, name, value)
            else:
                [setattr(sample, name, value) for sample in self.samples]

    def load_init(self):
        [sample.load_init() for sample in self.samples]

    def set_kernel(self, kernel):
        # this will call __setattr__ and change kernel for all samples
        self.kernel = kernel
        if self.gp_class == AdditiveStructureGPModel:
            self._orig_kernel = kernel

    @property
    def additive_structure(self) -> List[torch.Tensor]:
        if self.gp_class == AdditiveStructureGPModel:
            structure = [k.active_dims for k in self.kernel.kernels]
            return structure

    @property
    def is_restricted(self) -> bool:
        if self.gp_class == AdditiveStructureGPModel:
            return self.kernel != self._orig_kernel

    def restrict(self, active_group: int):
        if self.gp_class == AdditiveStructureGPModel:
            [sample.restrict(active_group) for sample in self.samples]

    def unrestrict(self):
        if self.gp_class == AdditiveStructureGPModel:
            [sample.unrestrict() for sample in self.samples]

    def clone(self):
        samples = self.samples
        self.samples = None
        new = copy.deepcopy(self)
        self.samples = samples
        new.samples = [sample.clone() for sample in self.samples]
        return new

    def _check_prior_setters(self):
        [sample._check_prior_setters() for sample in self.samples]
        return True
