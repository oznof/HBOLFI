import unittest

import torch
from torch.distributions import Uniform
from torch.distributions.constraints import _Real

from hbolfi.distributions import *
from hbolfi.distributions.inverse_gamma import InvGamma
from hbolfi.distributions.priors import Constraint
from hbolfi.utils import rng
from hbolfi import settings


STATE = torch.Generator().manual_seed(117).get_state()


class TestPrior(unittest.TestCase):
    def test_make_prior_on_class(self):
        sample_shape = (100,)
        with rng(STATE):
            a1 = torch.abs(torch.randn(4)) + 2
            b1 = a1 + 1.117 * torch.randn(4)
            a2 = torch.abs(torch.randn(4)) + 3
            b2 = a1 + 1.711 * torch.randn(4)
            init_p1, init_p2 = torch.abs(torch.randn(2, 4))
        u1 = InvGamma(a1, b1)
        u2 = InvGamma(a2, b2)
        p1 = InvGammaPrior(a1, b1, initial_value=init_p1)
        p2 = InvGammaPrior(a2, b2, initial_value=init_p2)
        with rng(STATE):
            samples_u1 = u1.sample(sample_shape)
            samples_u2 = u2.sample(sample_shape)
        with rng(STATE):
            samples_p1 = p1.sample(sample_shape)
            samples_p2 = p2.sample(sample_shape)
        self.assertTrue(p1.__class__ == p2.__class__)
        self.assertTrue(torch.equal(samples_u1, samples_p1))
        self.assertTrue(torch.equal(samples_u2, samples_p2))
        self.assertIsInstance(p1.constraint, Constraint)
        self.assertIsInstance(p2.constraint, Constraint)
        self.assertTrue(p1.constraint != p2.constraint)
        self.assertTrue(torch.equal(p1.constraint.initial_value, init_p1))
        self.assertTrue(torch.equal(p2.constraint.initial_value, init_p2))
        # supports are shared between instances of same class
        self.assertTrue(p1.support == p2.support)
        # support is a copy of the original
        self.assertTrue(p1.support != u1.support)

    def test_make_prior_on_instance_uniform(self):
        sample_shape = (100,)
        lb_u1, ub_u1 = torch.zeros(3), torch.ones(3)
        lb_u2, ub_u2 = -torch.ones(3), torch.zeros(3)
        init_u1 = torch.ones(3) / 2
        init_u2 = -torch.ones(3) / 3
        for i in range(2):
            with self.subTest(i=i):
                if i == 0:
                    u1 = Uniform(lb_u1, ub_u1)
                    u2 = Uniform(lb_u2, ub_u2)
                    p1 = make_prior(u1, initial_value=init_u1)
                    p2 = make_prior(u2, initial_value=init_u2)
                else:
                    spike_scale = torch.tensor([0.01])
                    spike1 = TruncatedNormal(loc=init_u1, scale=spike_scale, low=lb_u1, high=ub_u1)
                    slab1 = Uniform(low=lb_u1, high=ub_u1)
                    u1 = Mixture([spike1, slab1])
                    spike2 = TruncatedNormal(loc=init_u2, scale=spike_scale, low=lb_u2, high=ub_u2)
                    slab2 = Uniform(low=lb_u2, high=ub_u2)
                    u2 = Mixture([spike2, slab2])
                    p1 = SpikeUniformSlabPrior(mean=init_u1, low=lb_u1, high=ub_u1)
                    p2 = SpikeUniformSlabPrior(mean=init_u2, low=lb_u2, high=ub_u2)
                with rng(STATE):
                    samples_u1 = u1.sample(sample_shape)
                    samples_u2 = u2.sample(sample_shape)
                with rng(STATE):
                    samples_p1 = p1.sample(sample_shape)
                    samples_p2 = p2.sample(sample_shape)
                # prior class is not the same when calling make_prior on instances
                self.assertTrue(p1.__class__ != p2.__class__)
                self.assertTrue(torch.equal(samples_u1, samples_p1))
                self.assertTrue(torch.equal(samples_u2, samples_p2))
                self.assertIsInstance(p1.constraint, Constraint)
                self.assertIsInstance(p2.constraint, Constraint)
                self.assertTrue(p1.constraint != p2.constraint)
                self.assertTrue(torch.equal(p1.constraint.initial_value, init_u1))
                self.assertTrue(torch.equal(p2.constraint.initial_value, init_u2))
                # supports are not shared when calling make_prior on instances
                self.assertTrue(p1.support != p2.support)
                # support is a copy of the original
                self.assertTrue(p1.support != u1.support)
                self.assertTrue(p2.support != u2.support)
                self.assertTrue(torch.equal(p1.support.lower_bound, lb_u1))
                self.assertTrue(torch.equal(p1.support.upper_bound, ub_u1))
                self.assertTrue(torch.equal(p2.support.lower_bound, lb_u2))
                self.assertTrue(torch.equal(p2.support.upper_bound, ub_u2))

    def test_delta_priors(self):
        names = ('lengthscale', 'outputscale', 'noise', 'angular_weights', 'alpha', 'beta',
                 'quadpoly_a', 'quadpoly_b', 'quadpoly_c')
        with settings.fix_all_hyperparameters(state=True):
            prev_prior = None
            for name in names:
                prior = getattr(settings, name).prior()
                prior2 = getattr(settings, name).prior()
                self.assertIsInstance(prior, DeltaPrior)
                self.assertTrue(prior == prior2)
                # DeltaPrior has the same support for all hyperparameters, but objects are not the same
                self.assertIsInstance(prior.support, _Real)
                if prev_prior:
                    self.assertTrue(prior != prev_prior)
                    self.assertTrue(prior.support == prev_prior.support)
                    # constraint objects are tied to instances, despite support being shared
                    self.assertTrue(prior.constraint != prev_prior.constraint)
                prev_prior = prior

            init0 = torch.tensor(1.)
            init1 = torch.tensor(2.1234)
            init2 = torch.tensor(117.711)
            self.assertTrue(torch.equal(settings.lengthscale.initial_value, init0))
            self.assertTrue(torch.equal(settings.outputscale.initial_value, init0))
            o = settings.outputscale.prior()
            settings.lengthscale.set(prior='Delta', initial_value=init1)
            l = settings.lengthscale.prior()
            self.assertTrue(o.constraint != l.constraint)
            self.assertTrue(torch.equal(o.constraint.initial_value, init0))
            self.assertTrue(torch.equal(l.constraint.initial_value, init1))
            settings.outputscale.set(prior='Delta', initial_value=init2)
            o2 = settings.outputscale.prior()
            self.assertTrue(o != o2)
            self.assertTrue(torch.equal(o.constraint.initial_value, init0))
            self.assertTrue(torch.equal(l.constraint.initial_value, init1))
            self.assertTrue(torch.equal(o2.constraint.initial_value, init2))
            # check that the _prior_context class has not been changed
            self.assertTrue(settings._prior_context.initial_value is None)
            self.assertFalse(settings._prior_context._prior_registry)
            self.assertTrue(settings._prior_context._prior is None)


if __name__ == '__main__':
    unittest.main()