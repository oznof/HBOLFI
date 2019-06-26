from typing import Callable, Optional

from scipy.optimize import minimize

import torch
from torch.nn import Parameter
from torch.distributions import biject_to, constraints, LowerCholeskyTransform
from gpytorch.functions import add_jitter
from gpytorch.distributions import MultivariateNormal

from ..utils import hessian, rng
from ..distributions import SoftPlusTransform


class GaussianVI:
    def __init__(self,
                 log_prob: Callable[[torch.Tensor], torch.Tensor],
                 map_estimate: torch.Tensor,
                 num_dims: int,
                 num_groups: Optional[int] = None,
                 mode: str = 'oracle',
                 seed: int = 117,
                 with_mean: bool = False,
                 bounds: Optional[torch.Tensor] = None,
                 **kwargs):

        if num_groups is None:
            mode = 'diag'
        self.mode = mode
        self.log_prob = log_prob
        self.rng_state = torch.Generator().manual_seed(seed).get_state()
        self.num_dims = num_dims
        self.num_groups = num_groups
        self.dim_group = num_dims // num_groups
        # unconstrained to constrained
        self.transforms = {'chol': (SoftPlusTransform() if self.mode == 'diag' else LowerCholeskyTransform())}
        self.init_mean = map_estimate
        self.with_mean = with_mean
        if with_mean:
            self.transforms['mean'] = biject_to(constraints.interval(*bounds)),
            self.raw_mean = Parameter(self.transforms["mean"].inv(self.init_mean))
        x = self.init_mean.detach().clone().requires_grad_(True)
        h = hessian(lambda p: -log_prob(p), x)
        covar = torch.cholesky_inverse(torch.cholesky(h, upper=True), upper=True)
        self.init_covar = covar
        chol = torch.cholesky(covar, upper=False)
        if self.mode == 'oracle':
            ix = torch.arange(self.dim_group)
            chol_init = chol[(ix.unsqueeze(1), ix.unsqueeze(0))]
        elif mode == 'diag':
            chol_init = chol.diag()
        elif self.mode == 'full':
            chol_init = chol
        elif self.mode == 'partition':
            self.partition = kwargs['partition']
            chol_init = [chol[(group.unsqueeze(1), group.unsqueeze(0))] for group in self.partition]
            inv_chol_transform = self.transforms["chol"].inv
            self.raw_chol = Parameter(torch.cat([inv_chol_transform(l).reshape(-1) for l in chol_init]))
        else:
            raise NotImplementedError("Unknown mode.")
        if self.mode != 'partition':
            self.raw_chol = Parameter(self.transforms["chol"].inv(chol_init))

    def make_chol(self):
        transform = self.transforms["chol"]
        if self.mode == 'oracle':
            diag = torch.diag(torch.ones(self.num_groups))
            chol = transform(self.raw_chol)
            chol = torch.einsum("ab,cd->acbd", diag, chol).view(diag.size(0) * chol.size(0), diag.size(1) * chol.size(1))
        elif self.mode == 'diag':
            chol = transform(torch.diag(self.raw_chol))
        elif self.mode == 'partition':
            chol = torch.zeros(self.num_dims, self.num_dims)
            i = 0
            for group in self.partition:
                num_params = group.numel() ** 2
                chol[(group.unsqueeze(1), group.unsqueeze(0))] = self.raw_chol[i:i+num_params]
                i += num_params
        else:  # 'full'
            chol = transform(self.raw_chol)
        return chol

    def make_q(self):
        mean = self.transforms["mean"](self.raw_mean) if self.with_mean else self.init_mean
        chol = self.make_chol()
        covar = add_jitter(chol @ chol.t(), 1e-6)
        try:
            q = MultivariateNormal(mean, covar)
        except RuntimeError:
            q = MultivariateNormal(mean, add_jitter(covar, 1e-4))
        return q

    def optimize(self, num_samples, maxiter):

        def func_and_grad(raw):
            raw = torch.tensor(raw).reshape(self.raw_chol.shape).clone().requires_grad_(True)
            self.raw_chol = raw
            q = self.make_q()
            base_samples = torch.randn(num_samples, q.mean.size(-1))
            samples = q.rsample(base_samples=base_samples)
            vals = self.log_prob(samples)
            loss = -q.entropy() - torch.mean(vals)
            curr_loss = loss.item()
            loss.backward()
            # print(-torch.mean(vals))
            # print(f"loss={curr_loss}")
            # print(q.covariance_matrix)
            return [curr_loss, raw.grad.clone().numpy().flatten()]

        raw_param = self.raw_chol.detach().flatten()
        if self.with_mean:
            raw_param = torch.cat((self.raw_mean.detach(), raw_param))
        with rng(self.rng_state) as rs:
            res = minimize(func_and_grad, raw_param.numpy(), method='L-BFGS-B', jac=True,
                           options={"maxiter": maxiter}, bounds=None)
        self.rng_state = rs.generator_or_state
        raw_param = torch.tensor(res['x'])
        if self.with_mean:
            idx_split = self.init_mean.numel()
            self.raw_mean = raw_param[:idx_split]
            raw_param = raw_param[idx_split:]
        self.raw_chol = raw_param.reshape(self.raw_chol.shape)
        q = self.make_q()
        return q

    # def _optimize(self, num_samples, maxiter):
    #     optimizer = torch.optim.Adam([self.raw_chol], **{"lr": 0.001})
    #
    #     def closure():
    #         optimizer.zero_grad()
    #         q = self.make_q()
    #         samples = q.rsample([num_samples])
    #         loss = -q.entropy() - torch.mean(self.log_prob(samples))
    #         # print(loss.item())
    #         # print(q.covariance_matrix)
    #         # print(q.loc)
    #         loss.backward()
    #         return loss
    #
    #     prev_loss = None
    #     for _ in range(maxiter):
    #         curr_loss = optimizer.step(closure).item()
    #         ftol = abs(prev_loss - curr_loss) / max(1, abs(prev_loss), abs(curr_loss)) if prev_loss else 1
    #         if ftol <= 1e-9:
    #             break
    #         prev_loss = curr_loss
    #
    #     self.raw_chol.requires_grad = False
    #     #  self.raw_mean.requires_grad = False
    #     q = self.make_q()
    #     return q
