import time
import copy
import torch
import gpytorch

from hbolfi import AdditiveStructureGPModel, LCB, QuadPolyMean


class QuadFormModel:
    def __init__(self, kernel, num_dims, corr, search_space, mu_min=0.25,
                 mean_function=None, seed=117, copy_kernel=True, **kwargs):
        if not isinstance(kernel, gpytorch.kernels.AdditiveKernel):
            raise RuntimeError("QuadFormModel can only take in an AdditiveKernel")
        self.kernel = copy.deepcopy(kernel) if copy_kernel else kernel
        self.partition = [k.active_dims for k in kernel.kernels]
        self.num_dims = num_dims
        self.corr = corr
        self._build_corr_matrix()
        min_loc = kwargs.get('min_loc')
        self.min_loc = mu_min * torch.ones(num_dims) if min_loc is None else min_loc
        self.search_space = search_space
        if mean_function is None:
            self.mean_function = QuadPolyMean(self.num_dims)
            self.mean_function.initialize(a=self._a_weights, b=0, c=0)
            for p in self.mean_function.parameters():  # freeze parameters
                p.requires_grad = False
        self.seed = seed
        self.rng_obs = torch.Generator().manual_seed(self.seed).get_state()  # for generating noisy observations
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = kwargs.get('noise', 1e-2)
        for p in self.likelihood.parameters():
            p.requires_grad = False
        self.lengthscale = torch.zeros(self.num_dims)
        for k in self.kernel.kernels:
            self.lengthscale[k.active_dims] = k.lengthscale.detach().clone()
        self.outputscale = self.kernel.kernels[0].outputscale.detach().clone()
        try:
            # during init, this might not work for classes that overload the observe_noiseless method
            self.min_val = self.observe_noiseless(self.min_loc)
        except:
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

    def observe_noiseless(self, x: torch.Tensor):
        x = x.view(-1, self.num_dims)
        xdiff = x - self.min_loc
        f = 0.5 * torch.einsum('ni,id,nd->n', xdiff, self.Cinv, xdiff) - x.pow(2) @ self._a_weights
        f += self.mean_function(x)  # by default, mean_function is x.pow(2) @ self._a_weights
        return f

    def __call__(self, x: torch.Tensor):
        mean = self.observe_noiseless(x)
        covar = gpytorch.lazify(torch.zeros(mean.numel(), mean.numel()))
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def observe(self, x: torch.Tensor):
        rng_obs = torch.Generator().set_state(self.rng_obs)
        x = x.view(-1, self.num_dims)
        mvn = self.likelihood(self(x))
        base_samples = torch.randn(x.size(0), generator=rng_obs)
        y = mvn.sample(base_samples=base_samples)
        self.rng_obs = rng_obs.get_state()
        return y

    def __getstate__(self):
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class SemiparametricQuadFormModel(QuadFormModel):
    def __init__(self, kernel, num_dims, corr, search_space, mu_min=0.5,
                 radii=tuple([0.01, 0.1, 0.5, 1]), num_s=200,
                 mean_function=None, seed=117, copy_kernel=True,
                 **kwargs):
        # settings for generating data for the nonparametric part
        self.radii = radii
        self.num_s = num_s
        self.rng_data = torch.Generator().manual_seed(seed).get_state()
        super().__init__(kernel, num_dims, corr, search_space, mu_min, mean_function, seed, copy_kernel, **kwargs)
        # necessary to find optimum due to the nonparametric part of the model
        # it will be near self.min_loc
        self.acquisition = kwargs.get('acquisition', LCB)
        self.acquisition_kwargs = kwargs.get('acquisition_kwargs', {})
        self._build_model()

    def _build_model(self):
        xmin = torch.full((self.num_dims,), float('inf'))
        gpm = AdditiveStructureGPModel(kernel=self.kernel, train_inputs=None, train_targets=None,
                                       likelihood=gpytorch.likelihoods.GaussianLikelihood(
                                           noise_constraint=gpytorch.constraints.GreaterThan(1e-10)),
                                       mean=self.mean_function, optim_param=(),
                                       init={'lengthscale': self.lengthscale, 'outputscale': self.outputscale,
                                             'noise': 1e-10})
        gpm.likelihood.noise = 1e-10
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
        rng_data = torch.Generator().set_state(self.rng_data)
        x = torch.zeros((1 + len(self.radii) * self.num_s, self.num_dims))
        x[0] = self.min_loc
        for i, r in enumerate(self.radii):
            x_aux = torch.zeros(self.num_s, self.num_dims)
            for pid, pi in enumerate(self.partition):
                x_aux[:, pi] = torch.randn(self.num_s, len(pi), generator=rng_data)
                corr_group = self.C[(pi.unsqueeze(1), pi.unsqueeze(0))]
                x_aux[:, pi] = r * x_aux[:, pi] / torch.norm(x_aux[:, pi], dim=1).unsqueeze(1)  # sphere
                x_aux[:, pi] = x_aux[:, pi] @ torch.cholesky(corr_group, upper=True)
                #if i < 2:
                #    x_aux[:, pi] *= torch.rand(self.num_s, 1, generator=self.rng_data).pow(1./len(pi))  # ball
            x[1 + i * self.num_s:1 + (i + 1) * self.num_s] = x_aux + self.min_loc
        assert(self._within_bounds(x, slack=0.))
        # get noiseless observations
        f = super().observe_noiseless(x)
        self.rng_data = rng_data.get_state()
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
                raise KeyError
            return getattr(self.model, item)
