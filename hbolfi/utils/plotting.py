from typing import Callable, Sequence, Union, Optional

import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.cm
from matplotlib.colors import Colormap


def plot_contour(func: Callable[[torch.Tensor], torch.Tensor],
                 bounds: Sequence[float] = (-1., 1.),
                 ax: Optional[mpl.axes.Axes] = None,
                 num_points: int = 200,
                 levels: Optional[Union[int, Sequence, np.ndarray]] = None,
                 fill: bool = False,
                 colors: Union[str, Sequence[str]] = 'k',
                 cmap: Union[str, Colormap] = mpl.cm.jet,
                 take_exp: bool = False) -> mpl.axes.Axes:
    if ax is None:
        f, ax = plt.subplots(1, 1)
    grid_1d = np.linspace(*bounds, num_points)
    x, y = np.meshgrid(grid_1d, grid_1d)
    x_2d = np.vstack((x.flatten(), y.flatten())).T
    val = func(torch.as_tensor(x_2d)).numpy().reshape(x.shape)
    if take_exp:
        val = np.exp(val)
    if levels is None:
        levels_fn = np.logspace if take_exp else np.linspace
        levels = levels_fn(val.min(), val.max(), 100)
    if fill:
        ax.contourf(x, y, val, levels=levels, cmap=cmap)
    else:
        ax.contour(x, y, val, levels=levels, colors=colors)
    return ax
