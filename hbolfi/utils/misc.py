import torch


def normalize(x: torch.Tensor, mode: str = 'moment') -> torch.Tensor:
    x_shape = x.shape
    if x.ndim == 1:
        x = x.view(1, -1)
    if mode == 'moment':
        x_norm = x - x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        mask = std > 0
        x_norm[mask.expand(x_norm.shape)] /= std[mask]
    elif mode == 'maxmin':
        x_max = x.max(-1, keepdim=True)[0]
        x_min = x.min(-1, keepdim=True)[0]
        x_range = x_max - x_min
        x_norm = (x - x_min)
        mask = x_range > 0
        x_norm[mask.expand(x_norm.shape)] /= x_range[mask]
        x_norm = 2 * (x_norm - 0.5)
    else:
        raise ValueError("Mode not implemented. Choose 'moment' or 'maxmin'.")
    return x_norm.reshape(x_shape)
