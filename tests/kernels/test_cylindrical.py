import unittest
from itertools import product
from time import time

import numpy as np

import torch
import torch.distributions
import torch.distributions.constraints
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

import sampyl

from HyperSphere.GP.kernels.modules.radialization import RadializationKernel
from HyperSphere.feature_map.modules.kumaraswamy import Kumaraswamy
from HyperSphere.BO.shadow_inference.inference_sphere_origin import ShadowInference as inf_origin
from HyperSphere.GP.inference.inference import Inference as inf_regular
from HyperSphere.GP.models.gp_regression import GPRegression

import hbolfi
from hbolfi import CylindricalKernel, make_gp
from hbolfi import settings, rng
from hbolfi.simulators import Levy, Rosenbrock

SEED = 117


# HyperSphere computes the nll in a different way, see the comments in original_negative_log_likelihood
# here, we assume x_origin takes the same angle information as the point it is being compared to
# some changes have been made to the original nll function
def negative_log_likelihood(self, hyper=None):
    if hyper is not None:
        param_original = self.model.param_to_vec()
        self.cholesky_update(hyper)
    kernel_max = self.model.kernel.forward_on_identical().data[0]
    n_nonorigin = self.train_x_nonorigin_radius.size(0)
    one_radius = torch.autograd.Variable(torch.ones(1, 1)).type_as(self.train_x)
    # repeat by the number of origin points (1) not nonorigin
    K_non_ori_rel_radius = self.model.kernel.radius_kernel(self.train_x_nonorigin_radius,
                                                           one_radius * 0).repeat(1, self.train_y.numel() - n_nonorigin)
    # here it should take the diagonal, since at the origin the angle information is supposed to be the same as
    # the angles of nonorigin points
    K_non_ori_rel_sphere = self.model.kernel.sphere_kernel(self.train_x_nonorigin_sphere,
                                                           self.train_x_nonorigin_sphere).diag().reshape(-1, 1)
    K_non_ori_rel = K_non_ori_rel_radius * K_non_ori_rel_sphere
    chol_solver = torch.solve(torch.cat([self.mean_vec.index_select(0, self.ind_nonorigin), K_non_ori_rel], 1),
                              self.cholesky_nonorigin)[0]
    chol_solver_y = chol_solver[:, :1]  # L_11^-1 y1  (n_1 x 1)
    chol_solver_q = chol_solver[:, 1:]  # L_10 = L_11^-1 K_10  (n_1 x n_0)
    # same as before: number of origin points, do mm
    sol_p_sqr = kernel_max + self.model.likelihood(self.train_x_origin).repeat(self.train_y.numel() - n_nonorigin, 1) +\
                self.jitter - chol_solver_q.t().mm(chol_solver_q)  # (n_0 x n_0)
    # add checks:
    # terms are based on block matrix inversion
    term1 = - chol_solver_q.t().mm(chol_solver_y)  # -C A^-1 y_1
    if sol_p_sqr.numel() == 1:  # this should always be the case
        sol_p = torch.sqrt(sol_p_sqr.clamp(min=1e-12))
        term1 = term1 / sol_p  # -D^-1 C A^-1 y_1
        term2 = self.mean_vec.index_select(0, self.ind_origin).view(-1, 1) / sol_p  # D^-1 y_0
    else:
        sol_p = torch.cholesky(sol_p_sqr, upper=False)
        term1 = torch.solve(term1, sol_p)[0]  # -D^-1 C A^-1 y_1
        term2 = torch.solve(self.mean_vec.index_select(0, self.ind_origin).view(-1, 1), sol_p)[0]  # D^-1 y_0
    sol_y_0 = term1 + term2
    # remove mean function that was applied to sol_y_0
    nll = 0.5 * (torch.sum(chol_solver_y ** 2) + sol_y_0.t().mm(sol_y_0)) + torch.sum(
        torch.log(torch.diag(self.cholesky_nonorigin))) + torch.mean(torch.log(sol_p)) + 0.5 * self.train_y.size(
        0) * np.log(2 * np.pi)
    if hyper is not None:
        self.cholesky_update(param_original)
    return nll


# not described in the paper
# the way the nll is computed seems to be by taking an average of nlls, where in each nll the origin takes the
# same angle information as one point in the training data
# this is different than having x_origin have the same angle information as the point under comparison, which only
# requires a single computation of the nll.
def original_negative_log_likelihood(self, hyper=None):
    if hyper is not None:
        param_original = self.model.param_to_vec()
        self.cholesky_update(hyper)
    kernel_max = self.model.kernel.forward_on_identical().data[0]
    n_nonorigin = self.train_x_nonorigin_radius.size(0)
    # compute K_10, with 0 being expanded n_1 times -> (n_1 x n_1)
    one_radius = torch.autograd.Variable(torch.ones(1, 1)).type_as(self.train_x)
    K_non_ori_rel_radius = self.model.kernel.radius_kernel(self.train_x_nonorigin_radius,
                                                           one_radius * 0).repeat(1, n_nonorigin)
    K_non_ori_rel_sphere = self.model.kernel.sphere_kernel(self.train_x_nonorigin_sphere, self.train_x_nonorigin_sphere)
    K_non_ori_rel = K_non_ori_rel_radius * K_non_ori_rel_sphere
    chol_solver = torch.gesv(torch.cat([self.mean_vec.index_select(0, self.ind_nonorigin), K_non_ori_rel], 1),
                             self.cholesky_nonorigin)[0]
    chol_solver_y = chol_solver[:, :1]  # L_11^-1 y1  (n_1 x 1)
    chol_solver_q = chol_solver[:, 1:]  # L_10 = L_11^-1 K_10  (n_1 x n_1)
    sol_p_sqr = kernel_max + self.model.likelihood(self.train_x_origin).repeat(n_nonorigin, 1) + self.jitter - (
                chol_solver_q ** 2).sum(0).view(-1, 1)  # L_00 ^ 2, expanded to (n1 x 1)
    sol_p = torch.sqrt(sol_p_sqr.clamp(min=1e-12))
    # solve related to the origin.
    # (1) L_00^-1 y_0 - L_00^-1 L_01 L_11^-1 y1
    # since L_00, L_01 have been expanded -> (n1 x 1)
    # these are then averaged out in nll
    sol_y_i = (self.mean_vec.index_select(0, self.ind_origin) - chol_solver_q.t().mm(chol_solver_y)) / sol_p
    nll = 0.5 * (torch.sum(chol_solver_y ** 2) + torch.mean(sol_y_i ** 2)) + torch.sum(
        torch.log(torch.diag(self.cholesky_nonorigin))) + torch.mean(torch.log(sol_p)) + 0.5 * self.train_y.size(
        0) * np.log(2 * np.pi)
    if hyper is not None:
        self.cholesky_update(param_original)
    return nll


# inf_origin.negative_log_likelihood = original_negative_log_likelihood

# modified version of Hypersphere sampling where the priors are ignored
def sampling_no_priors(self, n_sample=10, n_burnin=100, n_thin=10):
    type_as_arg = list(self.model.likelihood.parameters())[0].data
    def logp(hyper):
        hyper_tensor = torch.from_numpy(hyper).type_as(type_as_arg)
        if self.model.out_of_bounds(hyper):
            return -np.inf
        self.model.vec_to_param(hyper_tensor)
        if not self.stable_parameters():
            return -np.inf
        prior = 0  # self.model.prior(hyper)
        likelihood = -self.negative_log_likelihood(hyper_tensor).item()
        return prior + likelihood
    hyper_numpy = self.model.param_to_vec().numpy()
    sampler = sampyl.Slice(logp=logp, start={'hyper': hyper_numpy}, compwise=True)
    samples = sampler.sample(n_burnin + n_thin * n_sample, burn=n_burnin + n_thin - 1, thin=n_thin)
    self.cholesky_update(torch.from_numpy(samples[-1][0]).type_as(type_as_arg))
    return torch.stack([torch.from_numpy(elm[0]) for elm in samples], 0).type_as(type_as_arg)


# modified version of gpytorch.mlls.ExactMarginalLogLikelihood.forward where priors are ignored
def mll_forward_no_priors(self, output, target, *params):
    output = self.likelihood(output, *params)
    res = output.log_prob(target)
    num_data = target.size(-1)
    return res.div_(num_data)


# modified version of BayesianGPModel.transform_jacobian (no op)
def transform_jacobian(model, params, transforms):
    return 0


class TestCylindricalKernel(unittest.TestCase):
    def setUp(self):
        if hasattr(torch, 'cdist'):
            delattr(torch, 'cdist')
        torch.set_default_dtype(torch.float32)
        gpytorch.settings._fast_covar_root_decomposition._state = False
        gpytorch.settings._fast_log_prob._state = False
        gpytorch.settings._fast_solves._state = False
        gpytorch.settings.fast_pred_var._state = False
        gpytorch.settings.fast_pred_samples._state = False
        gpytorch.settings.lazily_evaluate_kernels._state = False
        gpytorch.settings.num_trace_samples._global_value = 0
        self.generator = torch.Generator().manual_seed(SEED)

    def generate_data(self, dim, search_radius, n=10):
        with rng(self.generator):
            x = torch.randn(n, dim)
            r = x.norm(dim=-1, keepdim=True)
            mask_r = (r >= search_radius).squeeze()
            x[mask_r] = x[mask_r] / r[mask_r] * (search_radius * 0.99)
            self.assertTrue(torch.all(x.norm(dim=-1) < search_radius))
        return x

    def check_kernels(self, dim, outputscale, lengthscale, angular_weights, alpha=None, beta=None):
        search_radius = dim ** 0.5
        x1, x2 = [self.generate_data(dim, search_radius) for _ in range(2)]
        warping = False if alpha is None or beta is None else True
        radius_input_map = Kumaraswamy(ndim=1, max_input=search_radius) if warping else None
        max_power = angular_weights.numel() - 1
        k1 = RadializationKernel(max_power=max_power, search_radius=search_radius, radius_input_map=radius_input_map)
        if warping:
            param_vec = torch.log(torch.cat((outputscale, alpha, beta, lengthscale ** 2, angular_weights)))
        else:
            param_vec = torch.log(torch.cat((outputscale, lengthscale ** 2, angular_weights)))
        k1.vec_to_param(param_vec)
        gram_1 = k1(x1, x2)
        settings.lengthscale.set("Delta", lengthscale)
        settings.outputscale.set("Delta", outputscale)
        settings.angular_weights.set("Delta", angular_weights)
        if warping:
            settings.alpha.set("Delta", alpha)
            settings.beta.set("Delta", beta)
        k2 = CylindricalKernel(max_radius=search_radius, num_angular_weights=max_power + 1,
                               weight_normalization=True, warping=warping)
        gram_2 = k2(x1, x2)
        return torch.allclose(gram_1, gram_2)

    def qtest_cylindrical_kernel(self):
        dims = [4, 8, 12, 16, 20]
        outputscales = [torch.tensor([1.17]), torch.tensor([2.13])]
        lengthscales = [torch.tensor([1.17]), torch.tensor([2.13])]
        angular_weights = [torch.tensor([3.41, 5.14, 6.18, 3.13]), torch.tensor([1.17, 2.61, 0.97, 2.13])]
        # (None, None) and (1, 1) should be equivalent
        alphas = [None, torch.ones(1), 0.5 * torch.ones(1)]
        betas = [None, torch.ones(1), 2 * torch.ones(1)]
        configs = product(dims, outputscales, lengthscales, angular_weights, alphas, betas)
        for config in configs:
            dim, o, l, w, a, b = config
            with self.subTest(dim=dim, outputscale=o, lengthscale=l, angular_weights=w, alpha=a, beta=b):
                self.assertTrue(self.check_kernels(dim=dim, outputscale=o, lengthscale=l, angular_weights=w,
                                                   alpha=a, beta=b))

    def qtest_nll_origin(self):
        dim = 8
        origin = True
        reps = 10
        outputscale = torch.tensor([1.71])
        lengthscale = torch.tensor([2.31])
        angular_weights = torch.tensor([3.41, 5.14, 6.18, 3.13])
        alpha = torch.ones(1) * 0.75
        beta = torch.ones(1) * 1.75
        noise = torch.ones(1) * 0.45
        const = torch.zeros(1)
        f = Levy(dim)
        search_radius = dim ** 0.5
        for i in range(reps):
            with self.subTest(iteration=i):
                x = self.generate_data(dim, search_radius)
                inf_method = inf_regular
                if origin:
                    x[0] = 0.  # include origin in the train set
                    inf_method = inf_origin
                y = f.eval(x)
                radius_input_map = Kumaraswamy(ndim=1, max_input=search_radius)
                max_power = angular_weights.numel() - 1
                k1 = RadializationKernel(max_power=max_power, search_radius=search_radius, radius_input_map=radius_input_map)
                param_vec = torch.log(torch.cat((outputscale, alpha, beta, lengthscale ** 2, angular_weights)))
                k1.vec_to_param(param_vec)
                model = GPRegression(kernel=k1)
                model.likelihood.log_noise_var.data = torch.log(noise)
                model.mean.const_mean.data = const
                inference = inf_method((x, y), model)
                inference.cholesky_update(model.param_to_vec())
                nll1 = inference.negative_log_likelihood()

                settings.outputscale.set('Delta', outputscale)
                settings.lengthscale.set('Delta', lengthscale)
                settings.angular_weights.set('Delta', angular_weights)
                settings.alpha.set('Delta', alpha)
                settings.beta.set('Delta', beta)
                settings.noise.set('Delta', noise)
                settings.quadpoly_a.set('Delta', torch.zeros(1))
                settings.quadpoly_b.set('Delta', torch.zeros(1))
                settings.quadpoly_c.set('Delta', const)
                k2 = CylindricalKernel(active_dims=torch.arange(dim),
                                       max_radius=search_radius, num_angular_weights=max_power + 1,
                                       weight_normalization=True, warping=True)
                times = []
                for flag in [True, False]:
                    with self.subTest(original_training=flag), \
                         settings.cylindrical_original_training_strategy(flag):
                        model2 = make_gp(k2)
                        y = y.squeeze()

                        model2.set_train_data(inputs=x, targets=y, strict=False)
                        s = time()
                        # 2 alternative ways of doing the same, because priors are Deltas (+0)
                        nll2 = -model2.likelihood(model2(x)).log_prob(y)
                        mll = ExactMarginalLogLikelihood(model2.likelihood, model2)
                        nll2_alt = -mll(model2(x), y) * y.numel()
                        t = time()
                        times.append(t)
                # it is preferred to use the original training, not only because
                # it is more accurate, it also is faster
                        rtol = 1e-5 if flag else 1e-2
                        self.assertTrue(torch.allclose(nll1, nll2, rtol=rtol))
                        self.assertTrue(torch.allclose(nll2, nll2_alt))
                self.assertTrue(times[1] - times[0] > 0)

    def qtest_predict(self, verbose=False):
        dim = 20
        reps = 10
        outputscale = torch.tensor([1.71])
        lengthscale = torch.tensor([2.31])
        angular_weights = torch.tensor([3.41, 5.14, 6.18, 3.13])
        alpha = torch.ones(1) * 0.75
        beta = torch.ones(1) * 1.75
        noise = torch.ones(1) * 0.45
        const = torch.zeros(1)
        f = Levy(dim)
        search_radius = dim ** 0.5
        for i in range(reps):
            with self.subTest(iteration=i):
                x = self.generate_data(dim, search_radius, n=4)
                x[0] = 0.  # include origin in the train set
                inf_method = inf_origin
                y = f.eval(x)
                radius_input_map = Kumaraswamy(ndim=1, max_input=search_radius)
                max_power = angular_weights.numel() - 1
                k1 = RadializationKernel(max_power=max_power, search_radius=search_radius, radius_input_map=radius_input_map)
                param_vec = torch.log(torch.cat((outputscale, alpha, beta, lengthscale ** 2, angular_weights)))
                k1.vec_to_param(param_vec)
                model = GPRegression(kernel=k1)
                model.likelihood.log_noise_var.data = torch.log(noise)
                model.mean.const_mean.data = const
                inference = inf_method((x, y), model)
                inference.cholesky_update(model.param_to_vec())
                xtest = self.generate_data(dim, search_radius, n=3)
                s1 = time()
                pred_mean1, pred_var1 = inference.predict(xtest)[:2]
                t1 = time() - s1
                pred_mean1 = pred_mean1.squeeze()
                pred_var1 = pred_var1.squeeze()
                settings.outputscale.set('Delta', outputscale)
                settings.lengthscale.set('Delta', lengthscale)
                settings.angular_weights.set('Delta', angular_weights)
                settings.alpha.set('Delta', alpha)
                settings.beta.set('Delta', beta)
                settings.noise.set('Delta', noise)
                settings.quadpoly_a.set('Delta', torch.zeros(1))
                settings.quadpoly_b.set('Delta', torch.zeros(1))
                settings.quadpoly_c.set('Delta', const)
                k2 = CylindricalKernel(active_dims=torch.arange(dim),
                                       max_radius=search_radius, num_angular_weights=max_power + 1,
                                       weight_normalization=True, warping=True)
                model2 = make_gp(k2)
                y = y.squeeze()
                model2.set_train_data(inputs=x, targets=y, strict=False)
                model2.eval()  # set posterior mode
                settings.cylindrical_original_prediction_strategy._set_state(False)
                s2 = time()
                pred_mvn2 = model2(xtest)
                t2 = time() - s2
                pred_mean2 = pred_mvn2.mean
                pred_var2 = pred_mvn2.variance
                self.assertTrue(torch.allclose(pred_mean1, pred_mean2))
                self.assertTrue(torch.allclose(pred_var1, pred_var2))
                settings.cylindrical_original_prediction_strategy._set_state(True)
                s3 = time()
                pred_mvn3 = model2(xtest)
                t3 = time() - s3
                pred_mean3 = pred_mvn3.mean
                pred_var3 = pred_mvn3.variance
                self.assertTrue(torch.allclose(pred_mean1, pred_mean3))
                self.assertTrue(torch.allclose(pred_var1, pred_var3))
                if verbose:
                    print(f'mean diff (m2, ref): {torch.abs(pred_mean1 - pred_mean2).max():.3e}')
                    print(f'mean diff (m3, ref): {torch.abs(pred_mean1 - pred_mean3).max():.3e}')
                    print(f'var  diff (m2, ref): {torch.abs(pred_var1 - pred_var2).max():.3e}')
                    print(f'var  diff (m3, ref): {torch.abs(pred_var1 - pred_var3).max():.3e}')
                    print(f'time diff (m2, ref): {t2 - t1:.3e}')
                    print(f'time diff (m3, ref): {t3 - t1:.3e}')
                    print()

    def test_hyper_sampling(self):
        dim = 20
        f = Rosenbrock(dim)
        search_radius = dim ** 0.5
        x = self.generate_data(dim, search_radius, n=2)
        x[0] = 0.  # include origin in the train set
        y = f.eval(x)

        outputscale = torch.std(y).view(-1)
        lengthscale = torch.tensor([1.])
        angular_weights = torch.ones(4)
        noise = outputscale / 1000.
        const = torch.mean(y).view(-1)

        radius_input_map = None  # Kumaraswamy(ndim=1, max_input=search_radius)
        max_power = angular_weights.numel() - 1
        k1 = RadializationKernel(max_power=max_power, search_radius=search_radius, radius_input_map=radius_input_map)
        param_vec = torch.log(torch.cat((outputscale, lengthscale ** 2, angular_weights)))
        k1.vec_to_param(param_vec)
        model = GPRegression(kernel=k1)
        model.likelihood.log_noise_var.data = torch.log(noise)
        model.mean.const_mean.data = const
        inference = inf_origin((x, y), model)
        inference.cholesky_update(model.param_to_vec())
        print(model.param_to_vec())
        with rng(torch.Generator().manual_seed(SEED)):
            # gp_hyper_params = sampling_no_priors(inference, n_sample=10, n_burnin=0, n_thin=1)
            gp_hyper_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=1)

        # update constraint registry so that the parameters are in the log space
        @torch.distributions.biject_to.register(torch.distributions.constraints.positive)
        @torch.distributions.transform_to.register(torch.distributions.constraints.positive)
        def _transform_to_positive(constraint):
            return torch.distributions.ExpTransform()
        # constrained to be positive -> log space
        settings.outputscale.set("Uniform", outputscale)
        l_prior = hbolfi.make_prior(torch.distributions.Uniform(torch.tensor(1e-4), torch.tensor(2. * search_radius)),
                                    initial_value=search_radius)
        settings.lengthscale.set(l_prior, lengthscale)
        settings.angular_weights.set("HalfNormal", angular_weights)
        noise_prior = hbolfi.InvGammaPrior(0.001, 0.001, initial_value=noise)
        settings.noise.set(noise_prior, noise)
        #print(settings.noise.prior())
        # fixed parameters
        settings.quadpoly_a.set('Delta', torch.zeros(1))
        settings.quadpoly_b.set('Delta', torch.zeros(1))
        # real: no constraint
        settings.quadpoly_c.set("Normal", const)
        k2 = CylindricalKernel(active_dims=torch.arange(dim),
                               max_radius=search_radius, num_angular_weights=max_power + 1,
                               weight_normalization=True, warping=False)
        model2 = make_gp(k2, bayesian=True, sampler='slice',
                         sampler_options={'num_samples': 10, 'warmup_steps': 0, 'num_thinning': 1, 'num_chains': 1})
        #print(noise)
        #print(model2.likelihood.noise)

        #print(model2.likelihood.noise_covar.noise_prior)
        #print(settings.noise.prior())
        #import sys
        #sys.exit(1)
        y = y.squeeze()
        model2.set_train_data(inputs=x, targets=y, strict=False)
        orig_transform_jacobian = model2.transform_jacobian
        orig_exactmll_forward = gpytorch.mlls.ExactMarginalLogLikelihood.forward
        # model2.transform_jacobian = transform_jacobian
        # gpytorch.mlls.ExactMarginalLogLikelihood.forward = mll_forward_no_priors
        settings.initialize_before_fitting._state = False
        with rng(torch.Generator().manual_seed(SEED)):
            model2.fit()
        # revert changes
        model2.transform_jacobian = orig_transform_jacobian
        gpytorch.mlls.ExactMarginalLogLikelihood.forward = orig_exactmll_forward
        print("OUTPUTSCALE")
        print(gp_hyper_params[:, 0].exp())
        a = torch.cat([sample.kernel.radial_base_kernel.outputscale.data.view(-1) for sample in model2.samples])
        print(a)
        print("LENGTHSCALE")
        print(gp_hyper_params[:, 1].exp().sqrt())
        a = torch.cat([sample.kernel.radial_base_kernel.lengthscale.data.view(-1) for sample in model2.samples])
        print(a)
        print("ANGULAR WEIGHTS")
        print(gp_hyper_params[:, 2:6].exp())
        a = torch.stack([sample.kernel.angular_weights.data.view(-1) for sample in model2.samples])
        print(a)
        print("CONST")
        print(gp_hyper_params[:, 6])
        a = torch.cat([sample.mean.c.data.view(-1) for sample in model2.samples])
        print(a)
        print("NOISE")
        print(gp_hyper_params[:, 7].exp())
        a = torch.cat([sample.likelihood.noise.data.view(-1) for sample in model2.samples])
        print(a)
        print("========")
        print(model.likelihood.log_noise_var.exp())

if __name__ == '__main__':
    unittest.main()