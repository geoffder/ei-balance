from typing import Callable
import numpy as np


def merge(old, new):
    """Recursively merge nested dictionaries. Used to updating parameter
    dictionaries."""
    for k, v in new.items():
        if type(v) is dict:
            old[k] = merge(old[k], v)
        else:
            old[k] = v

    return old


def clean_axes(
    axes,
    remove_spines=["right", "top"],
    ticksize=11.0,
    spine_width=None,
    tick_width=None,
):
    """A couple basic changes I often make to pyplot axes. If input is an
    iterable of axes (e.g. from plt.subplots()), apply recursively."""
    if hasattr(axes, "__iter__"):
        for a in axes:
            clean_axes(
                a,
                remove_spines=remove_spines,
                ticksize=ticksize,
                tick_width=tick_width,
                spine_width=spine_width,
            )
    else:
        if spine_width is not None and tick_width is None:
            tick_width = spine_width
        for r in remove_spines:
            axes.spines[r].set_visible(False)
        axes.tick_params(
            axis="both",
            which="both",  # applies to major and minor
            **{r: False for r in remove_spines},  # remove ticks
            **{"label%s" % r: False for r in remove_spines},  # remove labels
            width=tick_width,
        )
        for ticks in axes.get_yticklabels():
            ticks.set_fontsize(ticksize)
        for ticks in axes.get_xticklabels():
            ticks.set_fontsize(ticksize)
        if spine_width is not None:
            for s in ["top", "bottom", "left", "right"]:
                axes.spines[s].set_linewidth(spine_width)


def find_bsln_return(
    arr, bsln_start=0, bsln_end=None, offset=0.0, step=10, pre_step=False
):
    idx = np.argmax(arr[bsln_end:]) + (0 if bsln_end is None else bsln_end)
    last_min = idx
    bsln = np.mean(arr[bsln_start:bsln_end]) + offset
    if step > 0:
        last = len(arr) - 1
        stop = lambda next: next <= last
        get_min = lambda i: np.argmin(arr[i : i + step]) + i
    else:
        stop = lambda next: next >= 0
        get_min = lambda i: np.argmin(arr[i + step : i]) + i + step

    while stop(idx + step):
        min_idx = get_min(idx)
        if arr[min_idx] < bsln:
            return last_min if pre_step else min_idx
        else:
            last_min = min_idx
            idx += step

    return min_idx


def find_rise_bsln(
    arr, bsln_start=0, bsln_end=None, offset=0.0, step=10, pre_step=False
):
    return find_bsln_return(arr, bsln_start, bsln_end, offset, -step, pre_step)


def exp_rise(x, m, tau):
    return m * (1 - np.exp(-x / tau))


def project_onto_line(line_a, line_b, pt):
    """Project a 2d point onto the line between a and b (len 2 ndarrays).
    Returns pair of the length along line starting from a and the projected
    coordinate."""
    a_b = line_b - line_a  # vector from a to b
    pt_a = pt - line_a  # vector from a to pt
    dp = a_b @ pt_a
    len_a_b = np.sqrt(a_b @ a_b)  # type:ignore
    len_pt_a = np.sqrt(pt_a @ pt_a)  # type:ignore
    cos = dp / (len_a_b * len_pt_a)  # type:ignore
    len_proj = cos * len_pt_a  # length along line that new point falls
    x = line_a[0] + len_proj * a_b[0] / len_a_b
    y = line_a[1] + len_proj * a_b[1] / len_a_b
    projected = np.array([x, y])
    return len_proj, projected


def n_digits(i):
    if i > 0:
        return int(np.log10(i)) + 1
    elif i == 0:
        return 1
    else:
        return int(np.log10(-i)) + 1


def norm_xcorr(a, b, mode="valid"):
    """Normalized 1d cross-correlation as in matlab
    https://stackoverflow.com/a/71005798"""
    return np.correlate(a / np.linalg.norm(a), b / np.linalg.norm(b), mode=mode)


def map_axis(f: Callable[[np.ndarray], np.ndarray], arr: np.ndarray, axis=-1):
    """Map the specified axis (defaults to -1) with the function `f`. `f` should
     thus expect an ndarray with the shape created by `axis` and all of its
     'sub-axes'. The shape returned by `f` may differ from the original shape of
    its input.

    For example:
    - `arr` of shape (2, 5) with `axis` = -1. f operates on a 1d array of length 5.
    - `arr` of shape (2, 5, 5) with `axis` = 1. f operates on an array of shape (5, 5).
    """
    in_shape = arr.shape
    if len(in_shape) > 1:
        reshaped = arr.reshape(np.prod(in_shape[:axis]).astype(int), *in_shape[axis:])
        mapped: np.ndarray = np.stack([f(a) for a in reshaped], axis=0)

        if len(mapped.shape) == 1:
            return mapped.reshape(in_shape[:axis])
        else:
            return mapped.reshape(*in_shape[:axis], *mapped.shape[1:])
    else:
        return f(arr)


def rolling_average(arr: np.ndarray, n=3, axis=-1) -> np.ndarray:
    """Same as moving average, but with convolve and same padding."""
    ones = np.ones(n)
    return map_axis(lambda a: np.convolve(a, ones, "same"), arr, axis=axis) / n

def avg_radians(thetas):
    s = 0
    c = 0
    n = 0
    for t in thetas:
        s += np.sin(t)
        c += np.cos(t)
        n += 1
    return np.arctan2(s / n, c / n)

def avg_degrees(degs):
    return avg_radians(map(np.deg2rad, degs))

def map_data(f, data):
    """Recursively apply the same operation to all ndarrays stored in the given
    dictionary. It may have arbitary levels of nesting, as long as the leaves are
    arrays and they are a shape that the given function can operate on."""

    def applyer(val):
        if type(val) == dict:
            return {k: applyer(v) for k, v in val.items()}
        else:
            return f(val)

    return {k: applyer(v) for k, v in data.items()}


def map2_data(f, d1, d2):
    def applyer(v1, v2):
        if type(v1) == dict:
            return {k: applyer(v1, v2) for (k, v1), v2 in zip(v1.items(), v2.values())}
        else:
            return f(v1, v2)

    if type(d1) == dict:
        return {k: applyer(v1, v2) for (k, v1), v2 in zip(d1.items(), d2.values())}
    else:
        return applyer(d1, d2)

def biexp(x, m, t1, t2, b):
    return m * (np.exp(-x / t1) - np.exp(-x / t2)) + b

def heaveside(x):
    if x < 0:
        return 0
    elif x == 0:
        return 0.5
    else:
        return 1

def alpha_fun(scale, tau):
    return np.vectorize(lambda x: scale * x / tau * np.exp(1 - (x / tau)) * heaveside(x))

def bialpha_fun(scale, tau1, tau2):
    scale = scale * tau1 / tau2
    return np.vectorize(lambda x: scale * x / tau1 * np.exp(1 - (x / tau2)) * heaveside(x))
