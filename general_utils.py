import numpy as np
import h5py as h5


def merge(old, new):
    """Recursively merge nested dictionaries. Used to updating parameter
    dictionaries."""
    for k, v in new.items():
        if type(v) is dict:
            old[k] = merge(old[k], v)
        else:
            old[k] = v

    return old


def clean_axes(axes, remove_spines=["right", "top"], ticksize=11.0):
    """A couple basic changes I often make to pyplot axes. If input is an
    iterable of axes (e.g. from plt.subplots()), apply recursively."""
    if hasattr(axes, "__iter__"):
        for a in axes:
            clean_axes(a, remove_spines=remove_spines, ticksize=ticksize)
    else:
        for r in remove_spines:
            axes.spines[r].set_visible(False)
        axes.tick_params(
            axis="both",
            which="both",  # applies to major and minor
            **{r: False for r in remove_spines},  # remove ticks
            **{"label%s" % r: False for r in remove_spines}  # remove labels
        )
        for ticks in axes.get_yticklabels():
            ticks.set_fontsize(ticksize)
        for ticks in axes.get_xticklabels():
            ticks.set_fontsize(ticksize)


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
    """Project a 2d point onto the line between a and b (len 2 ndarrays)."""
    a_b = line_b - line_a  # vector from a to b
    pt_a = pt - line_a  # vector from a to pt
    dp = a_b @ pt_a
    len_a_b = np.sqrt(a_b @ a_b)  # type:ignore
    len_pt_a = np.sqrt(pt_a @ pt_a)  # type:ignore
    cos = dp / (len_a_b * len_pt_a)  # type:ignore
    len_proj = cos * len_pt_a  # length along line that new point falls
    x = line_a[0] + len_proj * a_b[0] / len_a_b
    y = line_a[1] + len_proj * a_b[1] / len_a_b
    return np.array([x, y])
