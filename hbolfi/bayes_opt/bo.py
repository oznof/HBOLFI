from typing import Any, Callable, Dict, Optional, Type, Tuple, Union
from numbers import Number

import ghalton

import torch
from torch.nn import Module
from torch.quasirandom import SobolEngine
from torch.distributions import AffineTransform, ComposeTransform, SigmoidTransform
from torch.distributions import Distribution, Transform, Uniform
import gpytorch

from ..acquisitions import AcquisitionFunction, LCB, LCBAdd
from ..distributions import biject_to
from ..models import SurrogateModel
from ..utils import rng


class BayesianOptimization(Module):
    def __init__(self,
                 obj_function:        Callable[[torch.Tensor], torch.Tensor],
                 model:               SurrogateModel,
                 search_space:        Optional[Dict[str, Union[str, Union[str, Number, torch.Tensor, Transform]]]],
                 initial_samples:     int = 20,
                 train_inputs:        Optional[torch.Tensor] = None,
                 train_targets:       Optional[torch.Tensor] = None,
                 acquisition:         Type[AcquisitionFunction] = LCB,
                 acquisition_options: Optional[Dict[str, Any]] = None,
                 seed:                Optional[int] = None,
                 prior:               Optional[Distribution] = None,
                 set_cube_transform:  bool = False):
        super().__init__()
        self.obj_function = obj_function
        self.model = model
        if prior is None:
            if 'bounds' in search_space:
                lower_bound, upper_bound = search_space['bounds']
                prior = Uniform(lower_bound, upper_bound)
            else:
                raise RuntimeError(f"Specify a prior or bounds.")
        from_real = biject_to(prior.support)
        from_cube = ComposeTransform([SigmoidTransform().inv, from_real])
        search_space['from_real'] = from_real
        search_space['from_cube'] = from_cube
        self.prior = prior
        # this allows the model to apply a transformation to the inputs of the kernel
        if set_cube_transform and hasattr(model, 'to_centered_cube_transform'):
            transform = ComposeTransform([from_real.inv, SigmoidTransform(), AffineTransform(-1, 2, 0)])
            model.to_centered_cube_transform = transform
        self.transforms = {p: v for p, v in search_space.items() if p in ('from_cube', 'from_real')}
        self.search_space = search_space
        if isinstance(train_inputs, torch.Tensor) and isinstance(train_targets, torch.Tensor):
            assert(train_inputs.size(0) == train_targets.numel())
            self.model.set_train_data(inputs=train_inputs, targets=train_targets, strict=False)
        elif initial_samples > 0:
            train_inputs, train_targets = self.get_train_data(initial_samples)
            self.model.set_train_data(inputs=train_inputs, targets=train_targets, strict=False)
        self.seed = seed
        self.rng_state = (torch.get_rng_state() if self.seed is None else
                          torch.Generator().manual_seed(self.seed).get_state())
        if acquisition_options is None:
            acquisition_options = {}
        self.acquisition_options = acquisition_options
        self.acquisition = acquisition(self.model, self.search_space, seed=self.seed, **self.acquisition_options)
        self.eval()

    @property
    def dim(self):
        return self.model.train_inputs[0].size(-1)

    def get_train_data(self, initial_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if 'dim' in self.search_space:
            dim = self.search_space['dim']
        elif 'bounds' in self.search_space:
            dim = self.search_space['bounds'].size(1)
        else:
            raise RuntimeError("Unknown number of dimensions.")
        if dim <= 1111:
            inputs = SobolEngine(dim).draw(initial_samples)
        else:
            sequencer = ghalton.Halton(dim)
            inputs = sequencer.get(self.initial_samples)
        inputs = torch.as_tensor(inputs, dtype=torch.get_default_dtype()).view(-1, dim)
        train_inputs = self.transforms['from_cube'](inputs)
        train_targets = torch.tensor([self.obj_function(x) for x in train_inputs])
        return train_inputs, train_targets

    def step(self, acquire: bool = True, fit: bool = False,
             model_modifier: Optional[Callable[[], SurrogateModel]] = None,
             verbose: bool = False, linear_update: bool = False):
        if acquire:
            guess_point = self.model.train_inputs[0][torch.argmin(self.model.train_targets)]
            new_input, _ = self.acquisition.acquire(guess_point=guess_point, retain_graph=linear_update)
            new_target = torch.as_tensor([self.obj_function(new_input)])
            if not linear_update:
                train_inputs = torch.cat([self.model.train_inputs[0], new_input.view(1, -1)])
                train_targets = torch.cat([self.model.train_targets, new_target])
                self.model.set_train_data(inputs=train_inputs, targets=train_targets, strict=False)
            else:
                self.model = self.model.get_fantasy_model(new_input.view(1, -1), new_target.view(-1))
                self.acquisition.model = self.model  # sync model
        if fit or model_modifier:
            # some models rely on random sampling from priors (see ExactGP.load_init())
            with rng(self.rng_state) as rs:
                self.model.fit(verbose=verbose)
            self.rng_state = rs.generator_or_state
        if model_modifier:
            self.model = model_modifier(self.model)
            self.acquisition.model = self.model  # sync model

    def expected_min(self) -> Tuple[torch.Tensor, Union[float, torch.Tensor]]:
        acq_class = self.acquisition.__class__
        if acq_class not in [LCB, LCBAdd]:
            acq_class = LCB
        acquisition = acq_class(model=self.model, search_space=self.search_space, beta=torch.zeros(1),
                                seed=self.seed, **self.acquisition_options)
        guess_point = self.model.train_inputs[0][torch.argmin(self.model.train_targets)]
        with gpytorch.settings.skip_posterior_variances(True):
            xmin, valmin = acquisition.acquire(guess_point=guess_point)
        return xmin, valmin
