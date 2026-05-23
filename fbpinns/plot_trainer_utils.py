"""
Shared plotting utilities used by plot_trainer_2D and plot_trainer_3D
"""

import jax
import jax.numpy as jnp
import numpy as np


def _lim(v, factor=1.1):
    mi, ma = v.min(0), v.max(0)
    c = (mi+ma)/2
    w = factor*(ma-mi)/2
    if np.ndim(c) == 0:  # scalar
        vmax = max(abs(c-w), abs(c+w))
        return (-vmax, vmax)
    else:  # array (spatial coordinates)
        return (c-w, c+w)

def _plot_setup(x_batch_test, u_exact):
    # get general setup for plotting
    xlim, ulim = _lim(x_batch_test), _lim(u_exact)
    return xlim, ulim

def _to_numpy(f):
    # converts jnp arrays to np arrays
    def wrapper(*args):
        args = jax.tree_util.tree_map(lambda a: np.array(a) if isinstance(a, jnp.ndarray) else a, args)
        return f(*args)
    return wrapper
