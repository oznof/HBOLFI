import time
import copy

import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal

from .. import settings
from ..acquisitions import LCB
from ..distributions import DeltaPrior
from ..means import QuadPolyMean
from ..models import AdditiveStructureGPModel, SurrogateModel
from ..utils import rng


class QuadFormModel(SurrogateModel):
    def __init__(self, partition, num_dims, corr, search_space, mu_min=0.25,
                 mean_function=None, seed=None, **kwargs):
        super().__init__()
        delattr(self, 'train_inputs')
        delattr(self, 'train_targets')
        self.partition = partition
        self.num_dims = num_dims
        self.corr = corr
        self._build_corr_matrix()
        min_loc = kwargs.get('min_loc')
        self.min_loc = mu_min * torch.ones(num_dims) if min_loc is None else min_loc
        self.search_space = search_space
        if mean_function is None:
            a_prior = DeltaPrior(self._a_weights, initial_value=self._a_weights)
            b_prior = settings.quadpoly_b.prior('Delta')
            c_prior = settings.quadpoly_c.prior('Delta')
            mean_priors = {'quadpoly_a_prior': a_prior, 'quadpoly_b_prior': b_prior, 'quadpoly_c_prior': c_prior}
            self.mean_function = QuadPolyMean(self.num_dims, priors=mean_priors)
            for p in self.mean_function.parameters():  # freeze parameters
                p.requires_grad = False
        self.seed = seed
        if seed is None:
            seed = torch.get_rng_state()
        # for generating noisy observations
        self.rng_obs = torch.Generator().manual_seed(seed).get_state() if isinstance(seed, int) else seed
        noise = torch.as_tensor(kwargs.get('noise', 1e-2))
        noise_prior = DeltaPrior(noise, initial_value=noise)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior,
                                                                  noise_constraint=noise_prior.constraint)
        for p in self.likelihood.parameters():
            p.requires_grad = False
        try:
            # during init, this might not work for classes that overload the observe_noiseless method
            self.min_val = self.observe_noiseless(self.min_loc)
        except RuntimeError:
            self.min_val = None

    def _build_corr_matrix(self):
        self.C = torch.zeros(self.num_dims, self.num_dims)
        self.Cinv = torch.zeros(self.num_dims, self.num_dims)
        for pi in self.partition:
            d = len(pi)
            _corr = torch.full((d, d), self.corr)
            _corr[range(d), range(d)] = 1
            pi_idx = (pi.unsqueeze(1), pi.unsqueeze(0))
            self.C[pi_idx] = _corr
            self.Cinv[pi_idx] = torch.inverse(_corr)
        self._a_weights = 0.5 * self.Cinv[range(self.num_dims), range(self.num_dims)]

    def observe_noiseless(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.num_dims)
        xdiff = x - self.min_loc
        f = -x.pow(2).matmul(self._a_weights) + 0.5 * torch.einsum('ni,id,nd->n', [xdiff, self.Cinv, xdiff])
        f += self.mean_function(x)  # by default, mean_function is x.pow(2) @ self._a_weights
        return f

    def __call__(self, x: torch.Tensor) -> MultivariateNormal:
        mean = self.observe_noiseless(x)
        covar = gpytorch.lazify(torch.zeros(mean.numel(), mean.numel()))
        return MultivariateNormal(mean, covar)

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        with rng(self.rng_obs) as rs:
            x = x.view(-1, self.num_dims)
            mvn = self.likelihood(self(x))
            base_samples = torch.randn(x.size(0))
            y = mvn.sample(base_samples=base_samples)
        self.rng_obs = rs.generator_or_state
        return y

    def __getstate__(self):
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class SemiparametricQuadFormModel(QuadFormModel):
    def __init__(self, kernel, num_dims, corr, search_space, mu_min=0.25,
                 radii=tuple([0.01, 0.1, 0.2, 0.5]), num_s=200,
                 mean_function=None, seed=None, copy_kernel=True,
                 **kwargs):
        if not isinstance(kernel, gpytorch.kernels.AdditiveKernel):
            raise RuntimeError(f"{self.__class_.__name__} can only take in an AdditiveKernel")
        self.kernel = copy.deepcopy(kernel) if copy_kernel else kernel
        partition = [k.active_dims for k in kernel.kernels]
        super().__init__(partition, num_dims, corr, search_space, mu_min, mean_function, seed, **kwargs)
        # settings for generating data for the nonparametric part
        self.radii = radii
        self.num_s = num_s
        if seed is None:
            seed = torch.get_rng_state()
        self.rng_data = torch.Generator().manual_seed(seed).get_state() if isinstance(seed, int) else seed
        # necessary to find optimum due to the nonparametric part of the model
        # it will be near self.min_loc
        self.acquisition = kwargs.get('acquisition', LCB)
        self.acquisition_kwargs = kwargs.get('acquisition_kwargs', {})
        self._build_model()

    def _build_model(self):
        xmin = torch.full((self.num_dims,), float('inf'))
        noise = torch.tensor(1e-10)
        noise_prior = DeltaPrior(noise, initial_value=noise)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior,
                                                             noise_constraint=noise_prior.constraint)
        gpm = AdditiveStructureGPModel(kernel=self.kernel, train_inputs=None, train_targets=None,
                                       likelihood=likelihood, mean=self.mean_function)
        self.model = gpm
        for p in gpm.parameters():  # freeze all parameters
            p.requires_grad = False
        acquisition = self.acquisition(model=gpm, search_space=self.search_space, beta=torch.zeros(1), seed=self.seed,
                                       **self.acquisition_kwargs)
        while not self._within_bounds(xmin):
            x, y = self._sample_quadform_data()
            gpm.set_train_data(x, y, strict=False)
            gpm.eval()
            # minimize only the predictive mean
            with gpytorch.settings.skip_posterior_variances(True):
                start = time.time()
                xmin, valmin = acquisition.acquire(guess_point=self.min_loc)
                print(f"xmin: {xmin}, valmin: {valmin}")
                print(f"time to acquire: {time.time() - start}")
        self.min_loc = xmin
        self._min_val = valmin
        self.min_val = 0.

    def _within_bounds(self, x, slack=0.2):
        lower_bound, upper_bound = self.search_space['bounds']
        return torch.all(lower_bound + slack <= x) and torch.all(x <= upper_bound - slack)

    def _sample_quadform_data(self):
        # sample locations
        with rng(self.rng_data) as rs:
            x = torch.zeros((1 + len(self.radii) * self.num_s, self.num_dims))
            x[0] = self.min_loc
            for i, r in enumerate(self.radii):
                x_aux = torch.zeros(self.num_s, self.num_dims)
                for pid, pi in enumerate(self.partition):
                    x_aux[:, pi] = torch.randn(self.num_s, len(pi))
                    corr_group = self.C[(pi.unsqueeze(1), pi.unsqueeze(0))]
                    x_aux[:, pi] = r * x_aux[:, pi] / torch.norm(x_aux[:, pi], dim=1).unsqueeze(1)  # sphere
                    x_aux[:, pi] = x_aux[:, pi] @ torch.cholesky(corr_group, upper=True)
                    # if i < 2:
                    #     x_aux[:, pi] *= torch.rand(self.num_s, 1, generator=self.rng_data).pow(1./len(pi))  # ball
                x[1 + i * self.num_s:1 + (i + 1) * self.num_s] = x_aux + self.min_loc
            assert(self._within_bounds(x, slack=0.))
            # get noiseless observations
            f = super().observe_noiseless(x)
        self.rng_data = rs.generator_or_state
        return x, f

    def observe_noiseless(self, x: torch.Tensor):
        # given by the posterior predictive mean of a gaussian process regression model whose data have
        # been sampled from a quadratic form
        x = torch.as_tensor(x).view(-1, self.num_dims)
        with gpytorch.settings.skip_posterior_variances(True):
            f = self.model(x).mean - self._min_val
        return f

    def __getattr__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            if item == "model":
                raise RuntimeError
            return getattr(self.model, item)
