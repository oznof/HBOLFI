from typing import Dict, Optional, Union, Sequence
import math

import torch
from torch.distributions import Distribution

from .bo import BayesianOptimization
from ..distributions import biject_to
from ..models import SurrogateModel
from ..inference import map_estimate, laplace, nuts, GaussianVI


class BOLFI(BayesianOptimization):
    def unnormalized_loglikelihood(self, threshold: float, x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        model = self.model
        x = torch.as_tensor(x).view(-1, x.size(-1))
        predictive_mvn = model.likelihood(model(x))
        mean = predictive_mvn.mean
        var = predictive_mvn.variance
        z = (threshold - mean) / (var.sqrt() + eps)
        loglik = math.log(0.5) + torch.log(1 + torch.erf(z / math.sqrt(2)) + eps)
        return loglik

    def log_prob(self, threshold: float, x: torch.Tensor) -> torch.Tensor:
        """Unnormalized log posterior pdf evaluated at x."""
        logprob_prior = self.prior.log_prob(x).sum(-1).clamp(-1000., 0.)
        logprob = self.unnormalized_loglikelihood(threshold, x) + logprob_prior
        return logprob

    @staticmethod
    def potential_function(param: Union[Dict[str, torch.Tensor], torch.Tensor],
                           threshold: float, model: SurrogateModel,
                           prior: Optional[Distribution], eps: float = 1e-20) -> torch.Tensor:

        raw_x = param['raw_x'] if isinstance(param, dict) else param  # unconstrained
        if prior:
            transform = biject_to(prior.support)  # unconstrained -> constrained
            x = transform(raw_x)
        else:
            x = raw_x
        x = torch.as_tensor(x).view(-1, x.size(-1))
        predictive_mvn = model.likelihood(model(x))
        mean = predictive_mvn.mean
        var = predictive_mvn.variance
        z = (threshold - mean) / (var.sqrt() + eps)
        log_prob = math.log(0.5) + torch.log(1 + torch.erf(z / math.sqrt(2)) + eps)
        if prior:
            prior_log_prob = prior.log_prob(x).sum(-1).clamp(-1000., 0.)
            return -(log_prob + prior_log_prob + transform.log_abs_det_jacobian(raw_x, x).sum())[0]
        return -log_prob[0]

    def infer(self,
              threshold:      float,
              *args,
              method:         str = 'nuts',
              initial_params: Optional[torch.Tensor] = None,
              initialize_map: int = True,
              num_samples:    int = 5000,
              warmup_steps:   int = 250,
              num_chains:     int = 4,
              num_thinning:   int = 2,
              seed:           Optional[int] = None,
              with_prior:     bool = True,
              **kwargs) -> Union[Distribution, torch.Tensor, Sequence[torch.Tensor]]:

        seed = self.seed if seed is None else seed
        if 'dim' not in self.search_space:
            self.search_space['dim'] = self.dim
        log_prob = ((lambda p: self.log_prob(threshold, p)) if with_prior else
                    (lambda p: self.unnormalized_loglikelihood(threshold, p)))
        if method == 'nuts':
            num_samples = (num_samples * num_thinning) // num_chains
            if initialize_map:
                initial_params = map_estimate(log_prob, self.search_space, initial_params, seed=seed, **kwargs)
            if initial_params is not None:
                raw_initial_params = self.transforms['from_real'].inv(initial_params).detach()
                raw_initial_params = {'raw_x': raw_initial_params}
            else:
                raw_initial_params = None

            def potential_fn(param):
                return self.potential_function(param, threshold=threshold, model=self.model, prior=self.prior)

            res = nuts(potential_fn,
                       transforms={'raw_x': self.transforms['from_real'].inv},
                       num_samples=num_samples, initial_params=raw_initial_params,
                       num_chains=num_chains, warmup_steps=warmup_steps, num_thinning=num_thinning, seed=seed, **kwargs)
            samples, rhat, ess = res
            samples, rhat, ess = samples['raw_x'], rhat['raw_x'], ess['raw_x']
            return samples, rhat, ess
        elif method == 'laplace':
            return laplace(log_prob, self.search_space, initial_params, seed=seed, **kwargs)
        elif 'gaussian_vi_' in method:
            initial_params = map_estimate(log_prob, self.search_space, initial_params, seed=seed, **kwargs)
            mode = method.split('_')[-1]
            vi = GaussianVI(log_prob=log_prob, map_estimate=initial_params,
                            num_dims=self.search_space['dim'], mode=mode, seed=seed, **kwargs)
            return vi.optimize(num_samples, kwargs.get('maxiter'))
        else:
            raise NotImplementedError()
