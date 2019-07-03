from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import multiprocessing
import pyro
from pyro.infer.mcmc import NUTS, MCMC
from pyro.ops.stats import effective_sample_size, split_gelman_rubin


def nuts(potential_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
         transforms: Optional[Dict[str, torch.Tensor]] = None,
         num_samples: int = 5000,
         initial_params: Optional[Dict[str, torch.Tensor]] = None,
         num_chains: int = 1,
         warmup_steps: Optional[int] = None,
         num_thinning: int = 1,
         seed: Optional[int] = None,
         step_size: float = 1,
         adapt_step_size: bool = True,
         adapt_mass_matrix: bool = True,
         full_mass: bool = False,
         use_multinomial_sampling: bool = True,
         jit_compile: bool = False,
         jit_options: Optional[dict] = None,
         ignore_jit_warnings: bool = False,
         target_accept_prob: float = 0.8,
         max_tree_depth: int = 10,
         mp_context: Optional[str] = None,
         disable_progbar: bool = False,
         **kwargs) -> Union[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                            Tuple[Dict[str, torch.Tensor]]]:
    nuts_kernel = NUTS(model=None, potential_fn=potential_fn, transforms=transforms,
                       step_size=step_size, adapt_step_size=adapt_step_size,
                       adapt_mass_matrix=adapt_mass_matrix, full_mass=full_mass,
                       use_multinomial_sampling=use_multinomial_sampling,
                       jit_compile=jit_compile, jit_options=jit_options, ignore_jit_warnings=ignore_jit_warnings,
                       target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    if initial_params is not None:
        nuts_kernel.initial_params = initial_params
    if num_chains == 1:
        if seed is not None:
            pyro.set_rng_seed(seed)
    else:
        if seed is not None:
            torch_initial_seed_func = torch.initial_seed
            torch.initial_seed = lambda: seed  # required for parallel workers
        if multiprocessing.get_sharing_strategy() != 'file_system':
            multiprocessing.set_sharing_strategy("file_system")
    mcmc_run = MCMC(nuts_kernel, warmup_steps=warmup_steps, num_samples=num_samples,
                    num_chains=num_chains, mp_context=mp_context, disable_progbar=disable_progbar).run()
    if num_chains > 1 and seed is not None:
        torch.initial_seed = torch_initial_seed_func
    samples_dict, ess_dict, rhat_dict = {}, {}, {}
    get_exec_traces = kwargs.pop('get_exec_traces')
    if get_exec_traces:
        return mcmc_run.exec_traces
    # must provide one of the two
    param_names = transforms.keys() if transforms else initial_params.keys()
    for param_name in param_names:
        samples = torch.stack([trace[param_name] for trace in mcmc_run.exec_traces])
        if num_chains == 1:
            samples = samples.unsqueeze(0)
        samples = samples[torch.tensor(mcmc_run._idx_by_chain)]
        samples = samples[:, ::num_thinning, :]
        ess = effective_sample_size(samples)
        rhat = split_gelman_rubin(samples)
        samples = samples.reshape(-1, samples.size(-1))
        samples_dict[param_name] = samples
        ess_dict[param_name] = ess
        rhat_dict[param_name] = rhat
    return samples_dict, ess_dict, rhat_dict
