import warnings
import math

import torch
import gpytorch
from gpytorch.constraints import Interval

from .. import settings
from ..distributions import TrainingDistribution


# gpytorch.module.Module.sample_from_prior does not take into consideration the sample shape
def sample_from_prior(self, prior_name):
    (*submodule_names, prior_name) = prior_name.split('.')
    module = self
    for submodule_name in submodule_names:
        module = getattr(module, submodule_name)
    """Sample parameter values from prior. Modifies the module's parameters in-place."""
    if prior_name not in module._priors:
        raise RuntimeError("Unknown prior name '{}'".format(prior_name))
    prior, closure, setting_closure = module._priors[prior_name]
    if setting_closure is None:
        raise RuntimeError("Must provide inverse transform to be able to sample from prior.")
    sample_shape = closure().shape
    if prior.event_shape:
        sample_shape = sample_shape[:-len(prior.event_shape)]
    if prior.batch_shape:
        sample_shape = sample_shape[:-len(prior.batch_shape)]
    setting_closure(prior.sample(sample_shape))


def register_constraint(self, param_name, constraint, replace=True):
    if param_name not in self._parameters:
        raise RuntimeError("Attempting to register constraint for nonexistent parameter.")
    constraint_name = param_name + "_constraint"
    if constraint_name in self._constraints:
        current_constraint = self._constraints[constraint_name]
    else:
        current_constraint = None
    if isinstance(current_constraint, Interval) and not replace:
        new_constraint = constraint.intersect(current_constraint)
    else:
        new_constraint = constraint
    self.add_module(constraint_name, new_constraint)
    self._constraints[constraint_name] = new_constraint
    # re-initialize the parameter if the constraint specifies an initial value
    if new_constraint.initial_value is not None:  # CHANGED: initialize the constrained value
        self.initialize(**{param_name.replace('raw_', ''): new_constraint.initial_value})


def _extract_named_priors(module, memo=None, prefix=""):
    if memo is None:
        memo = set()
    if hasattr(module, "_priors"):
        for name, (prior, closure, inv_closure) in module._priors.items():
            if prior is not None and (prior not in memo or 'kernels' in prefix):
                memo.add(prior)
                full_name = ("." if prefix else "").join([prefix, name])
                yield full_name, prior, closure, inv_closure
    for mname, module_ in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + mname
        for name, prior, closure, inv_closure in _extract_named_priors(module_, memo=memo, prefix=submodule_prefix):
            yield name, prior, closure, inv_closure


def _load_raw_parameters(self, **kwargs):
    for name, val in kwargs.items():
        if '.' in name:
            module, name = self._get_module_and_name(name)
            module._load_raw_parameters(**{name: val})
        else:
            delattr(self, name)
            setattr(self, name, val)


def gaussian_likelihood_set_priors(self, priors=None, with_default=True, without_initial_value=False):
    if priors is None:
        priors = {}
    if with_default or 'noise_prior' in priors:
        prior = priors.get('noise_prior', settings.noise.prior())
        module = self.noise_covar
        module.register_prior('noise_prior', prior, lambda: module.noise, lambda v: module._set_noise(v))
        if without_initial_value:
            init_value = prior.constraint.initial_value
            prior.constraint.initial_value = None
        module.register_constraint('raw_noise', prior.constraint)
        if without_initial_value:
            prior.constraint.initial_value = init_value
    return self


def gaussian_likelihood_call(self, input, *params, **kwargs):
    if isinstance(input, TrainingDistribution):
        return input
    return gpytorch.likelihoods.Likelihood.__call__(self, input, *params, **kwargs)


# hack: allow other general training distributions through a mock distribution, TrainingDistribution
def exact_mll_forward(self, output, target, *params):
    if not isinstance(output, torch.distributions.MultivariateNormal) and \
       not isinstance(output, TrainingDistribution):
        raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

    # Get the log prob of the marginal distribution
    output = self.likelihood(output, *params)
    res = output.log_prob(target)

    # Add terms for SGPR / when inducing points are learned
    added_loss = torch.zeros_like(res)
    for added_loss_term in self.model.added_loss_terms():
        added_loss = added_loss.add(added_loss_term.loss())

    res = res.add(0.5, added_loss)

    # Add log probs of priors on the (functions of) parameters
    for name, prior, closure, _ in self.named_priors():
        res.add_(prior.log_prob(closure()).sum())

    # Scale by the amount of data we have
    num_data = target.size(-1)
    return res.div_(num_data)


# the problem appears in inv_matmul when trying to compute second order derivatives
# for now, bypass it by calling torch.solve, which uses LU decomposition
# make sure to disable gpytorch.settings.fast_computations.solves, otherwise inv_matmul is still used
def exact_predictive_covar(self, test_test_covar, test_train_covar):
    if gpytorch.settings.fast_pred_var.on():
        self._last_test_train_covar = test_train_covar

    if gpytorch.settings.skip_posterior_variances.on():
        return gpytorch.lazy.ZeroLazyTensor(*test_test_covar.size())

    if gpytorch.settings.fast_pred_var.off():
        dist = self.train_prior_dist.__class__(
            torch.zeros_like(self.train_prior_dist.mean),
            self.train_prior_dist.lazy_covariance_matrix,
        )
        if gpytorch.settings.detach_test_caches.on():
            train_train_covar = self.likelihood(dist, self.train_inputs).lazy_covariance_matrix.detach()
        else:
            train_train_covar = self.likelihood(dist, self.train_inputs).lazy_covariance_matrix

        test_train_covar = gpytorch.delazify(test_train_covar)
        train_test_covar = test_train_covar.transpose(-1, -2)
        if gpytorch.settings.fast_computations.solves.off():
            covar_correction_rhs = torch.solve(train_test_covar, gpytorch.delazify(train_train_covar))[0]
        else:
            covar_correction_rhs = train_train_covar.inv_matmul(train_test_covar)
        # For efficiency
        if torch.is_tensor(test_test_covar):
            # We can use addmm in the 2d case
            if test_test_covar.dim() == 2:
                return gpytorch.lazify(torch.addmm(1, test_test_covar, -1, test_train_covar, covar_correction_rhs))
            else:
                return gpytorch.lazify(test_test_covar + test_train_covar @ covar_correction_rhs.mul(-1))
        # In other cases - we'll use the standard infrastructure
        else:
            return test_test_covar + gpytorch.lazy.MatmulLazyTensor(test_train_covar, covar_correction_rhs.mul(-1))

    precomputed_cache = self.covar_cache
    covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache,
                                                                               test_train_covar)
    if torch.is_tensor(test_test_covar):
        return gpytorch.lazify(
            torch.add(test_test_covar, -1, covar_inv_quad_form_root @ covar_inv_quad_form_root.transpose(-1, -2))
        )
    else:
        return test_test_covar + gpytorch.lazy.MatmulLazyTensor(
            covar_inv_quad_form_root, covar_inv_quad_form_root.transpose(-1, -2).mul(-1)
        )


def interval_transform(self, tensor):
    """
    Transforms a tensor to satisfy the specified bounds.

    If upper_bound is finite, we assume that `self.transform` saturates at 1 as tensor -> infinity. Similarly,
    if lower_bound is finite, we assume that `self.transform` saturates at 0 as tensor -> -infinity.

    Example transforms for one of the bounds being finite include torch.exp and torch.nn.functional.softplus.
    An example transform for the case where both are finite is torch.nn.functional.sigmoid.
    """
    if not self.enforced:
        return tensor
    if gpytorch.settings.debug.on():
        max_bound = torch.max(self.upper_bound)
        min_bound = torch.min(self.lower_bound)

        if max_bound == math.inf or min_bound == -math.inf:
            raise RuntimeError(
                "Cannot make an Interval directly with non-finite bounds. Use a derived class like "
                "GreaterThan or LessThan instead."
            )
    scale = self.upper_bound - self.lower_bound
    transformed_tensor = (self._transform(tensor) * scale) + self.lower_bound
    return transformed_tensor


def interval_inverse_transform(self, transformed_tensor):
    """
    Applies the inverse transformation.
    """
    if not self.enforced:
        return transformed_tensor

    if gpytorch.settings.debug.on():
        max_bound = torch.max(self.upper_bound)
        min_bound = torch.min(self.lower_bound)

        if max_bound == math.inf or min_bound == -math.inf:
            raise RuntimeError(
                "Cannot make an Interval directly with non-finite bounds. Use a derived class like "
                "GreaterThan or LessThan instead."
            )
    scale = self.upper_bound - self.lower_bound
    tensor = self._inv_transform((transformed_tensor - self.lower_bound) / scale)
    return tensor


def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        # TODO: Remove once fixed in pytorch (#16780)
        if A.dim() > 2 and A.is_cuda:
            if torch.isnan(L if out is None else out).any():
                raise RuntimeError

        return L
    except RuntimeError as e:
        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        jitter_ori = jitter
        print(A)
        print("===========")
        print(A.diag())
        for i in range(5):
            print(f"GOING IN jitter={jitter_ori * (10 ** i)}")
            jitter = jitter_ori * (10 ** i)
            Aprime = A.clone()
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter)
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                # TODO: Remove once fixed in pytorch (#16780)
                if A.dim() > 2 and A.is_cuda:
                    if torch.isnan(L if out is None else out).any():
                        raise RuntimeError("singular")
                warnings.warn(f"A not p.d., added jitter of {jitter} to the diagonal", RuntimeWarning)
                return L
            except RuntimeError:
                continue
        raise e
