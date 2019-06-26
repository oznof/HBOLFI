from collections import OrderedDict
from typing import Sequence, Union, Optional
import math

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP


def partition_to_assignment(partition: Sequence[Union[torch.Tensor, list, tuple]]) -> torch.Tensor:
    z = torch.empty(int(math.fsum([len(group) for group in partition])))
    for v, p in enumerate(partition):
        z[p] = v
    return z


def partition_metrics(partition_ref: Sequence[Union[torch.Tensor, list, tuple]],
                      partition: Sequence[Union[torch.Tensor, list, tuple]],
                      eps: float = 1e-20,
                      res_dict: Optional[OrderedDict] = None):
    a = partition_to_assignment(partition_ref)
    b = partition_to_assignment(partition)
    n = a.numel()
    assert(b.numel() == n)
    tp = eps
    tn = eps
    pred_p = eps
    for i in range(n-1):
        pred_p += torch.sum(b[i] == b[i+1:]).item()
        tp += torch.sum((a[i] == a[i+1:]) & (b[i] == b[i+1:])).item()
        tn += torch.sum((a[i] != a[i+1:]) & (b[i] != b[i+1:])).item()
    total_pairs = n * (n-1) / 2
    rand_idx = (tp + tn) / total_pairs
    truth_p = math.fsum([(a[i] == a[i+1:]).sum() for i in range(n-1)]) + eps
    truth_n = math.fsum([(a[i] != a[i+1:]).sum() for i in range(n-1)]) + eps
    tpr = tp / truth_p
    tnr = tn / truth_n
    precision = tp / pred_p
    f1_score = 2 / (1/tpr + 1 / precision)
    keys = ('rand_idx', 'tpr', 'tnr', 'precision', 'f1_score')
    values = (rand_idx, tpr, tnr, precision, f1_score)
    if res_dict is None:
        res_dict = OrderedDict()
    res_dict.update(OrderedDict(zip(keys, values)))
    return res_dict


def summary_stats(model: ExactGP, test_inputs: torch.Tensor, test_targets: torch.Tensor,
                  test_llik_ref: Optional[torch.Tensor] = None, res_dict: Optional[OrderedDict] = None,
                  preffix: str = '', suffix: str = ''):
    llik = model.log_likelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    test_mvn = model(test_inputs)
    disc_error = torch.norm(test_mvn.mean - test_targets, p=1, dim=-1).mean().item()
    test_llik = mll(test_mvn, test_targets).item()
    keys = ['llik', 'disc_error', 'test_llik']
    values = [llik, disc_error, test_llik]
    if test_llik_ref is not None:
        keys.append('test_llik_ratio')
        values.append(test_llik - test_llik_ref)
    keys = (preffix + k + suffix for k in keys)
    if res_dict is None:
        res_dict = OrderedDict()
    res_dict.update(OrderedDict(zip(keys, values)))
    return res_dict



