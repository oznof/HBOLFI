from typing import List, Type, Tuple, Optional
import time
import math
from math import log
import copy

import torch
from gpytorch.kernels import Kernel, AdditiveKernel

from .mcmc import MCMCStructureSelector
from ...models import AdditiveStructureGPModel
from ...utils import rng


class MHProposal:
    pass


class SplitMergeProposal(MHProposal):
    """
    Transition kernel for Metropolis Hastings sampler.
    """

    @staticmethod
    def split_kernel_random(kernel_list: List[Kernel]) -> Tuple[List[Kernel], float, float]:
        singleton_kernels = [kernel for kernel in kernel_list if kernel.active_dims.numel() == 1]
        non_singleton_kernels = [kernel for kernel in kernel_list if kernel.active_dims.numel() > 1]
        if len(non_singleton_kernels) == 0:
            return singleton_kernels, 0., 0.
        rand_ind = torch.randint(len(non_singleton_kernels), (1,)).item()
        kernel_to_split = non_singleton_kernels[rand_ind]
        old_active_dims = kernel_to_split.active_dims
        while True:  # make sure there is a split
            split_vec = torch.rand(old_active_dims.numel()) < 0.5
            if torch.sum(split_vec) != 0 and torch.sum(1 - split_vec) != 0:
                break
        ard = bool(kernel_to_split.ard_num_dims)
        active_dims1 = old_active_dims[split_vec]
        kernel_cls = kernel_to_split.__class__
        kernel1 = kernel_cls(active_dims=active_dims1, ard_num_dims=active_dims1.numel() if ard else None)
        active_dims2 = old_active_dims[1 - split_vec]
        kernel2 = kernel_cls(active_dims=active_dims2, ard_num_dims=active_dims2.numel() if ard else None)
        # prob split:
        # 1/|non_singleton_kernels| * 1/(2^(|dims_in_splitted_kernel|-1) - 1)
        proposal_logprob = - log(len(non_singleton_kernels)) - log(2**(old_active_dims.numel() - 1) - 1)
        non_singleton_kernels.pop(rand_ind)
        new_kernel_list = non_singleton_kernels + singleton_kernels + [kernel1, kernel2]
        # prob merge:
        # 1 / from len(new_kernel_list) choose 2
        kk = len(new_kernel_list)
        reverse_logprob = - log(kk) - log(kk - 1) + log(2)
        return new_kernel_list, proposal_logprob, reverse_logprob

    @staticmethod
    def merge_kernel_random(kernel_list: List[Kernel]) -> Tuple[List[Kernel], float, float]:
        if len(kernel_list) == 1:
            return kernel_list, 0., 0.
        rand_ind1 = torch.randint(len(kernel_list), (1,)).item()
        remaining = list(range(len(kernel_list)))
        remaining.pop(rand_ind1)
        rand_ind2 = remaining[torch.randint(len(remaining), (1,)).item()]
        kk = len(kernel_list)
        proposal_logprob = - log(kk) - log(kk - 1) + log(2)
        old_kernels = [kernel_list[rand_ind1], kernel_list[rand_ind2]]
        new_active_dims = torch.sort(torch.cat([k.active_dims for k in old_kernels]))[0]
        ard = bool(old_kernels[0].ard_num_dims) or bool(old_kernels[1].ard_num_dims)
        kernel_cls = old_kernels[0].__class__
        new_kernel = kernel_cls(active_dims=new_active_dims, ard_num_dims=new_active_dims.numel() if ard else None)
        new_kernel_list = [v for i, v in enumerate(kernel_list) if i not in [rand_ind1, rand_ind2]]
        new_kernel_list.append(new_kernel)
        non_singleton_kernels = [kernel for kernel in new_kernel_list if len(kernel.active_dims) > 1]
        reverse_logprob = -log(len(non_singleton_kernels)) - log(2**(new_active_dims.numel() - 1) - 1)
        return new_kernel_list, proposal_logprob, reverse_logprob

    @staticmethod
    def propose(kernel_list: List[Kernel]) -> Tuple[List[Kernel], float, float]:
        non_singleton_kernels = [kernel for kernel in kernel_list if kernel.active_dims.numel() > 1]
        if len(non_singleton_kernels) == 0:
            out = SplitMergeProposal.merge_kernel_random(kernel_list)
        elif len(kernel_list) == 1:
            out = SplitMergeProposal.split_kernel_random(kernel_list)
        elif torch.rand(1).item() < 0.5:
            out = SplitMergeProposal.merge_kernel_random(kernel_list)
        else:
            out = SplitMergeProposal.split_kernel_random(kernel_list)
        return out


class MHStructureSelector(MCMCStructureSelector):
    """
    Metropolis-Hastings sampler for non-overlapping additive decompositions.
    """
    def __init__(self, *args, proposal: Optional[Type[MHProposal]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.proposal = SplitMergeProposal if not proposal else proposal

    def sample(self) -> Tuple[AdditiveStructureGPModel, float, float]:
        new_model = copy.deepcopy(self.current_model)
        new_kernel_list, proposal_logprob, reverse_logprob = self.proposal.propose(list(new_model.kernel.kernels))
        new_model.set_kernel(AdditiveKernel(*new_kernel_list))
        new_model.optimize()
        return new_model, proposal_logprob, reverse_logprob

    def run(self, iters: int = 1) -> AdditiveStructureGPModel:
        with rng(self.rng_state) as rs:
            for i in range(iters):
                new_model, proposal_logprob, reverse_logprob = self.sample()
                new_loglik = new_model.log_likelihood
                top = new_loglik + reverse_logprob
                bot = self.current_model.log_likelihood + proposal_logprob
                alpha = min(1., math.exp(top - bot))
                accept = torch.rand(1).item() <= alpha
                if self.verbose:
                    print(f"iter={i:5} prob={alpha:.3f} accept={str(accept):5}\n"
                          f"\tproposal={new_model.additive_structure}")
                if accept:
                    self.current_model = new_model
        self.rng_state = rs.generator_or_state
        if self.save_trace:
            self.record(self.current_model)
        return self.current_model
