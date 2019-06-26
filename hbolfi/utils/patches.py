import math
import torch
import gpytorch
from gpytorch.constraints import Interval


# gpytorch.module.Module.sample_from_prior does not take into consideration the sample shape
# fix this by including closure().shape
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
    setting_closure(prior.sample(closure().shape))


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


# the problem appears in inv_matmul when trying to compute second order derivatives
# for now, bypass it by calling torch.solve, which uses LU decomposition
# make sure to disable gpytorch.settings.fast_computations.solves, otherwise inv_matmul is still used
def exact_predictive_covar(self, test_test_covar, test_train_covar):
    """
    Computes the posterior predictive covariance of a GP

    Args:
        test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs
        test_test_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test inputs

    Returns:
        :obj:`gpytorch.lazy.LazyTensor`: A LazyTensor representing the predictive posterior covariance of the
                                           test points
    """
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
