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


# Copied from https://docs.astropy.org/en/stable/api/astropy.stats.circcorrcoef.html
def circcorrcoef(alpha, beta, axis=None, weights_alpha=None, weights_beta=None):
    """Computes the circular correlation coefficient between two array of
    circular data.

    Parameters
    ----------
    alpha : ndarray or `~astropy.units.Quantity`
        Array of circular (directional) data, which is assumed to be in
        radians whenever ``data`` is ``numpy.ndarray``.
    beta : ndarray or `~astropy.units.Quantity`
        Array of circular (directional) data, which is assumed to be in
        radians whenever ``data`` is ``numpy.ndarray``.
    axis : int, optional
        Axis along which circular correlation coefficients are computed.
        The default is the compute the circular correlation coefficient of the
        flattened array.
    weights_alpha : numpy.ndarray, optional
        In case of grouped data, the i-th element of ``weights_alpha``
        represents a weighting factor for each group such that
        ``sum(weights_alpha, axis)`` equals the number of observations.
        See [1]_, remark 1.4, page 22, for detailed explanation.
    weights_beta : numpy.ndarray, optional
        See description of ``weights_alpha``.

    Returns
    -------
    rho : ndarray or `~astropy.units.Quantity` ['dimensionless']
        Circular correlation coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.stats import circcorrcoef
    >>> from astropy import units as u
    >>> alpha = np.array([356, 97, 211, 232, 343, 292, 157, 302, 335, 302,
    ...                   324, 85, 324, 340, 157, 238, 254, 146, 232, 122,
    ...                   329])*u.deg
    >>> beta = np.array([119, 162, 221, 259, 270, 29, 97, 292, 40, 313, 94,
    ...                  45, 47, 108, 221, 270, 119, 248, 270, 45, 23])*u.deg
    >>> circcorrcoef(alpha, beta) # doctest: +FLOAT_CMP
    <Quantity 0.2704648826748831>

    References
    ----------
    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".
       Series on Multivariate Analysis, Vol. 5, 2001.
    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from 'Topics in
       Circular Statistics (2001)'". 2015.
       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>
    """
    if np.size(alpha, axis) != np.size(beta, axis):
        raise ValueError("alpha and beta must be arrays of the same size")

    mu_a = circmean(alpha, axis, weights_alpha)
    mu_b = circmean(beta, axis, weights_beta)

    sin_a = np.sin(alpha - mu_a)
    sin_b = np.sin(beta - mu_b)
    rho = np.sum(sin_a * sin_b) / np.sqrt(np.sum(sin_a * sin_a) * np.sum(sin_b * sin_b))

    return rho
