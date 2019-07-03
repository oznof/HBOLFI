import math
from typing import List, Dict, Optional
import copy

import torch
from gpytorch.models.gp import GP
from gpytorch.kernels import Kernel, AdditiveKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..surrogate_model import SurrogateModel
from .exact_gp import ExactGPModel
from .additive_gp import AdditiveStructureGPModel
from ...inference import nuts
from ...distributions import DeltaPrior


class BayesianGPModel(GP, SurrogateModel):
    def __init__(self, kernel: Kernel, *args,
                 sampler: str = 'nuts',
                 sampler_options: Optional[dict] = None,
                 **kwargs):
        self.__dict__['_initialized'] = False
        super().__init__()
        delattr(self, 'train_inputs')
        delattr(self, 'train_targets')
        self.gp_class = AdditiveStructureGPModel if isinstance(kernel, AdditiveKernel) else ExactGPModel
        self.samples = [self.gp_class(kernel, *args, **kwargs)]
        self.sampler = sampler
        if sampler_options is None:
            sampler_options = {'num_samples': 10, 'warmup_steps': 20, 'num_thinning': 1, 'num_chains': 1}
        self.sampler_options = sampler_options if sampler_options else {}
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
        covar = torch.stack([mvn.covariance_matrix for mvn in mvns])
        return MultivariateNormal(mean, covar)

    @staticmethod
    def transform_jacobian(model: ExactGPModel, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        r = 0
        for name, param in params.items():
            constraint = model.constraint_for_parameter_name(name)
            if constraint is not None:
                r = r + constraint.log_abs_det_jacobian(param).sum()
        return r

    @staticmethod
    def potential_function(params: Dict[str, torch.Tensor], model: ExactGPModel,
                           mll: ExactMarginalLogLikelihood) -> torch.Tensor:
        # TODO: Catch exception
        # try:
        model._load_raw_parameters(**params)
        output = model(model.train_inputs[0])
        log_joint = mll(output, model.train_targets)
        log_joint = log_joint + BayesianGPModel.transform_jacobian(model, params)
        return -log_joint
        # except RuntimeError:  # return `nan` instead of raise Value Error if not be able to get Cholesky
        #     r = 0
        #     for p in params.values():
        #         r = r + 0 * p.sum()
        #     return torch.tensor(float('nan')) + r

    def fit(self, train_inputs: Optional[torch.Tensor] = None,
            train_targets: Optional[torch.Tensor] = None, verbose=False):
        base_sample = self.samples[0]
        base_sample.load_init()
        if train_inputs is not None and train_targets is not None:
            base_sample.set_train_data(inputs=train_inputs, targets=train_targets, strict=False)
        mll = ExactMarginalLogLikelihood(base_sample.likelihood, base_sample)
        if self.sampler == 'nuts':

            def potential_fn(param):
                return BayesianGPModel.potential_function(param, model=base_sample, mll=mll)

            param_state = {}
            for name, prior, _, _ in self.named_priors():
                if not isinstance(prior, DeltaPrior):
                    name = name.replace('_prior', '')
                    (*submodules, name) = name.split('.')
                    pname = 'raw_' + name
                    module = base_sample
                    for submodule in submodules:
                        module = getattr(module, submodule)
                    param_state['.'.join(submodules + [pname])] = getattr(module, pname)
            param_states = nuts(potential_fn=potential_fn, initial_params=param_state,
                                get_exec_traces=True, **self.sampler_options)
            self.samples = [base_sample.clone()._load_raw_parameters(**state) for state in param_states]

    def __getattr__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            return getattr(self.samples[0], item)

    def __setattr__(self, name, value):
        if not self._initialized:
            self.__dict__[name] = value
        else:
            if name in self.__dict__:
                self.__dict__[name] = value
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
