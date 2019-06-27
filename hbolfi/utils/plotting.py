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
                 minmax_scale: bool = False,
                 **kwargs) -> mpl.axes.Axes:
    if ax is None:
        f, ax = plt.subplots(1, 1)
    grid_1d = np.linspace(*bounds, num_points)
    x, y = np.meshgrid(grid_1d, grid_1d)
    x_2d = np.vstack((x.flatten(), y.flatten())).T
    val = func(torch.as_tensor(x_2d)).numpy().reshape(x.shape)
    if minmax_scale:
        valmin = val.min()
        valmax = val.max()
        val = (val - valmin) / (valmax - valmin)
    if kwargs.get('level_fn'):
        levels = kwargs.pop('level_fn')(val.min(), val.max())
    elif levels is None:
        levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
        if not minmax_scale:
            levels *= val.max()
    if fill:
        ax.contourf(x, y, val, levels=levels, cmap=cmap, **kwargs)
    else:
        ax.contour(x, y, val, levels=levels, colors=colors, **kwargs)
    return ax
