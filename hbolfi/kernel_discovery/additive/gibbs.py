from typing import Tuple, Optional
from math import log
from time import time

import torch
from torch.distributions import Gumbel
from gpytorch.kernels import AdditiveKernel

from .mcmc import MCMCStructureSelector
from ...models import AdditiveStructureGPModel
from ...utils import rng


# TODO: save models to tempdir and load them if the number of dimensions is high
class GibbsStructureSelector(MCMCStructureSelector):
    """
    Gibbs sampler for non-overlapping additive decompositions.
    """

    def __init__(self, model: AdditiveStructureGPModel, *args, iters_as_llik_eval: bool = True,
                 dim_limit: Optional[int] = None,  **kwargs):
        super().__init__(model, *args, **kwargs)
        self.iters_as_llik_eval = iters_as_llik_eval
        self.max_llik_evals = None  # this is updated in each call to run, if not iters_as_llik_eval it doesnt matter
        if dim_limit:
            assert(dim_limit <= model.train_inputs[0].shape[-1])
        self.dim_limit = dim_limit

    def sample(self) -> Tuple[AdditiveStructureGPModel, int]:
        dim = self.current_model.train_inputs[0].shape[-1]
        curr_model = self.current_model
        llik_evals = 0
        components = torch.randperm(dim)  # shuffled dimensions
        for d in components:
            partition = curr_model.additive_structure
            z = torch.empty(dim)
            for ii, dd in enumerate(partition):
                z[dd] = ii
            len_group = torch.sum(z == z[d]).item()  # len of group that includes d
            singleton_model = curr_model if len_group == 1 else None  # model with singleton group d
            log_prob = torch.zeros(dim)
            llik = torch.zeros(dim)
            new_models = []
            z0 = int(z[d].item())  # original assignment
            for m in range(dim):
                z[d] = m
                active_dims_plus = torch.nonzero(z == m).squeeze(-1)
                len_group = active_dims_plus.numel()
                # avoid unnecessary model optimizations
                if m == z0:  # original assignment
                    new_model = curr_model
                elif len_group == 1 and singleton_model is not None:
                    new_model = singleton_model  # equivalent models
                elif self.dim_limit is not None and len_group > self.dim_limit:
                    new_model = None  # if partition exceeds dim_limit
                    llik[m] = -float('inf')
                else:
                    # build model
                    new_model = curr_model.clone()
                    kernel_list = list(new_model.kernel.kernels)
                    kernel_minus = kernel_list[z0]  # kernel from which d must be removed
                    kernel_plus = kernel_list[m] if m < len(kernel_list) else None  # kernel to which d must be added
                    if kernel_plus:
                        ard = bool(kernel_plus.ard_num_dims)
                        kernel_plus = kernel_plus.__class__(active_dims=active_dims_plus,
                                                            ard_num_dims=active_dims_plus.numel() if ard else None)
                        kernel_list[m] = kernel_plus
                    else:  # singleton kernel
                        ard = bool(kernel_minus.ard_num_dims)
                        kernel_plus = kernel_minus.__class__(active_dims=active_dims_plus,
                                                             ard_num_dims=active_dims_plus.numel() if ard else None)
                        kernel_list.append(kernel_plus)
                    active_dims_minus = torch.nonzero(kernel_minus.active_dims != d).squeeze()
                    if active_dims_minus.numel() > 0:
                        ard = bool(kernel_minus.ard_num_dims)
                        kernel_minus = kernel_minus.__class__(active_dims=active_dims_minus,
                                                              ard_num_dims=active_dims_minus.numel() if ard else None)
                        kernel_list[z0] = kernel_minus
                    else:  # singleton kernel_minus must be removed
                        kernel_list.pop(z0)
                    new_model.set_kernel(AdditiveKernel(*kernel_list))
                    new_model.fit()
                    llik_evals += 1
                    if len_group == 1:
                        singleton_model = new_model
                if self.dim_limit is None or len_group <= self.dim_limit:
                    llik[m] = new_model.log_likelihood
                log_prob[m] = llik[m] + log(len_group - 1 + 1)
                new_models.append(new_model)
            m = torch.argmax(log_prob + Gumbel(0, 1).sample((dim,))).item()
            curr_model = new_models[m]
            if self.iters_as_llik_eval and llik_evals >= self.max_llik_evals:
                break
        return curr_model, llik_evals

    def run(self, model: Optional[AdditiveStructureGPModel] = None, iters: int = 1) -> AdditiveStructureGPModel:
        if model is not None:
            self.current_model = model
        if self.verbose:
            print(f"In {self.__class__.__name__}")
        i = 0
        start0 = time()
        with rng(self.rng_state) as rs:
            while iters > 0:
                start1 = time()
                self.max_llik_evals = iters
                # in each call, llik_evals is always at least the number of evals
                # that guarantees a single component update
                # this number is <= dim, varies according to the current partition
                new_model, llik_evals = self.sample()
                if self.verbose:
                    print(f"\titer={i:5} time={time() - start1:.3f} structure={new_model.additive_structure}")
                    i += 1
                self.current_model = new_model
                iters -= (llik_evals if self.iters_as_llik_eval else 1)
        self.rng_state = rs.generator_or_state
        if self.save_trace:
            self.record(self.current_model)
        if self.verbose:
            print(f"Out {self.__class__.__name__}, total time={time() - start0:.3f}")
        return self.current_model
