import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import h5py as h5
import json

from analysis import calc_tuning
from general_utils import unpack_hdf, clean_axes
from modelUtils import scale_180_from_360


def vonmises_fit(pref):
    """Returns vonmises fitting function with given preferred direction.
    Signature matches that expected by scipy.optimize.curve_fit."""
    def fit(theta, peak, width):
        """Fiscella et al., 2015.
        Visual coding with a population of direction-selective neurons.
        Expects theta in degrees (for now)."""
        return peak * np.exp((1 / width ** 2) * np.cos(np.deg2rad(theta - pref)))
    return fit


def gauss_fit(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_fit_line(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - y) ** 2) / np.sum(y))
    return fit_line(x, y, gauss_fit, p0=[1, mean, sigma])


def fit_line(x, y, func, pts=100, **fit_kwargs):
    """Fit data to given function and return function mapped onto the linear
    space between the lower and upper ranges of the x data. y can be be a
    1d ndarray, or 2d of shape (N, x.size)."""
    flat_y = y.reshape(-1)
    flat_x = np.repeat(x, y.size // x.size)
    pars, _ = curve_fit(f=func, xdata=flat_x, ydata=flat_y, **fit_kwargs)
    x_ax = np.linspace(np.min(x), np.max(x), pts)
    return x_ax, func(x_ax, *pars)


def thresholded_area(vm, thresh):
    """Take Vm ndarray (with time as last (or only) axis) and return
    with time dimension reduced to the area above the given threshold."""
    orig_shape = vm.shape
    reshaped = vm.copy().reshape(-1, orig_shape[-1])  # flatten non-T dims
    reshaped[reshaped < thresh] = thresh
    areas = np.sum((reshaped - thresh), axis=1)
    if len(orig_shape) > 1:
        return areas.reshape(*orig_shape[:-1])
    else:
        return np.squeeze(areas)


def thresh_area_tuning(data, thresh=-55):
    return {
        k: thresholded_area(d["soma"]["Vm"], thresh)
        for k, d in data.items()
    }


def plot_basic_ds_metrics(theta_diffs, dsis, thetas):
    """Plot E-I theta difference against DSis and preferred thetas in adjacent
    subplots."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 7))

    axes[0].scatter(theta_diffs, dsis)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("DSi", fontsize=13)

    axes[1].scatter(theta_diffs, thetas)
    axes[1].set_ylim(0, 180)
    axes[1].set_ylabel("Preferred Theta", fontsize=13)
    axes[1].set_xlabel("E-I Angle Difference", fontsize=13)

    clean_axes(axes)
    return fig


def plot_fancy_scatter_ds_metrics(theta_diffs, dsis, thetas):
    """Plot DSis vs preferred thetas in a single plot with theta difference
    represented as colour."""
    fig, ax = plt.subplots(1)
    scatter = ax.scatter(thetas, dsis, c=theta_diffs, cmap="jet")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Preferred Theta", fontsize=13)
    ax.set_ylabel("DSi", fontsize=13)

    ax.set_title("E-I Theta Difference -> Colour")
    fig.colorbar(scatter, ax=ax, orientation="vertical", pad=.1)
    return fig


def plot_raw_metric(dirs, base_metric, diff_keys=["0.0", "90.0", "180.0"],
                    relative_dirs=True, show_trials=True, fit="gauss"):
    """Plot response amplitude against stimulus direction for the given list
    of theta difference conditions."""
    fig, axes = plt.subplots(1, len(diff_keys), sharey=True, figsize=(10, 4))
    dir_ax = dirs
    dir_label = "%sDirection" % ("Relative " if relative_dirs else "")

    for ax, diff in zip(axes, diff_keys):
        diff_float = float(diff)
        if relative_dirs:
            dir_ax = scale_180_from_360(dirs - diff_float)
            diff_float = 0.  # correct for fit function

        if show_trials:
            for trial in base_metric[diff]:
                ax.scatter(dir_ax, trial, marker="x", c="0", alpha=.2)
        mean = np.mean(base_metric[diff], axis=0)
        ax.scatter(dir_ax, mean, c="r", s=45)

        if fit == "gauss":
            fit_x, fit_y = gauss_fit_line(dir_ax, mean)
        else:
            fit_x, fit_y = fit_line(dir_ax, mean, vonmises_fit(diff_float))
        ax.plot(fit_x, fit_y, c="0", linestyle="--")

        ax.set_xlim(-190, 145)
        ax.set_xlabel(dir_label, fontsize=13)
        ax.set_title("E-I Diff = %s" % diff)

    axes[0].set_ylabel("Response", fontsize=13)

    clean_axes(axes)
    fig.tight_layout()
    return fig


def plot_peak_responses(base_metric):
    """Plot maximal response amplitude against theta difference.
    NOTE: Undecided what the best metric / manipulation to display is yet, and
    how many panels to do. Options include:
      -- strongest responding direction amplitude (average over trials first)
      -- max response across all trials / directions
      -- max response for each trial (across all directions)
      -- any more?
    """
    fig, ax = plt.subplots(1)
    theta_diffs = np.array([float(t) for t in base_metric.keys()])
    # average over trials, then take the strongest responding direction
    prefs = [np.max(np.mean(rs, axis=0)) for rs in base_metric.values()]
    # maxs = [np.max(rs) for rs in base_metric.values()]

    ax.scatter(theta_diffs, prefs)
    ax.set_ylim(0)
    ax.set_ylabel("Max Response", fontsize=13)
    ax.set_xlabel("E-I Angle Difference", fontsize=13)

    clean_axes(ax)
    return fig


def trial_by_trial_tuning(dirs, base_metric):
    """Calculate DSi and preferred theta metrics for each trial for each
    theta difference condition in the base_metric input dict, which takes the
    form: {"0.0": ndarray of shape (trials, dirs), ...}"""
    trial_tuning = {}
    for k, v in base_metric.items():
        dsis, thetas = calc_tuning(v, dirs, dir_ax=1)
        trial_tuning[k] = {"DSis": dsis, "thetas": thetas}
    return trial_tuning


def avg_raw_tuning(dirs, base_metric):
    """Calculate average tuning by first calculating the average response with
    the given base_metric (dict of diff string -> ndarray shape (trials, dirs)).
    """
    metrics = [
        calc_tuning(np.mean(v, axis=0, keepdims=True), dirs, dir_ax=1)
        for v in base_metric.values()
    ]
    # unpack zipped list of tuples into two ndarrays
    dsis, thetas = [np.concatenate(tup) for tup in zip(*metrics)]
    return dsis, thetas


def avg_tuning_metrics(tuning_dict):
    """Calculate average tuning by averaging tuning metrics (dsi and theta) for
    individual trials. Input dict has theta difference keys and values in the
    form of {"DSis": ndarray (1D), "thetas": ndarray (1D)}."""
    avg_dsis = [np.mean(v["DSis"]) for v in tuning_dict.values()]
    avg_thetas = [np.mean(v["thetas"]) for v in tuning_dict.values()]
    return np.concatenate(avg_dsis), np.concatenate(avg_thetas)


def get_traces(data):
    return { k: d["soma"]["Vm"] for k, d in data.items() }


def plot_traces(dirs, traces, diff_keys=["0.0", "90.0", "180.0"]):
    timeax = np.arange(traces[diff_keys[0]].shape[-1]) * .1
    fig, axes = plt.subplots(
        len(diff_keys), len(dirs), figsize=(16, 10),
        sharex="col", sharey="row", squeeze=False,
    )

    vmin, vmax = np.inf, -np.inf
    for diff, row in zip(diff_keys, axes):
        dir_avgs = np.mean(traces[diff], axis=0)
        vmin = min(vmin, np.min(dir_avgs))
        vmax = max(vmax, np.max(dir_avgs))
        for i, (avg, ax) in enumerate(zip(dir_avgs, row)):
            ax.plot(timeax[1000:], avg[1000:])

    # shared Y (row) settings
    for diff, row in zip(diff_keys, axes):
        row[0].set_ylim(vmin, vmax)
        row[0].set_ylabel("%s° Θ Diff (mV)" % diff, size=14)

    # shared X (column) settings
    for direction, col_top, col_bot in zip(dirs, axes[0], axes[-1]):
        col_top.set_title("%i°" % direction, fontsize=18)
        col_bot.set_xlabel("Time (ms)", size=14)

    # axis clean-up
    for row in axes:
        for i, a in enumerate(row):
            if i > 0:
                a.spines["left"].set_visible(False)
                a.tick_params(left=False)  # don't draw ticks on ghost spine
            else:
                for ticks in (a.get_yticklabels()):
                    ticks.set_fontsize(11)
            a.spines["right"].set_visible(False)
            a.spines["top"].set_visible(False)

    fig.tight_layout()

    return fig


# TODO: peak / max response for each E-I difference (good to see inflection
# point of when the the angles are far enough apart to allow maximal preferred
# directtion response)


if __name__ == "__main__":
    basest = "/mnt/Data/NEURONoutput/tuning/"
    fig_path = basest + "figs/"

    with h5.File(basest + "theta_diff_exp.h5", "r") as f:
        all_data = unpack_hdf(f)

    theta_diffs = np.array([float(t) for t in all_data.keys()])
    dirs = json.loads(all_data["0.0"]["params"])["dir_labels"]
    dirs_180 = np.vectorize(scale_180_from_360)(dirs)

    thresh_areas = thresh_area_tuning(all_data, thresh=-55)
    tuning = trial_by_trial_tuning(dirs, thresh_areas)
    # avg_dsis, avg_thetas = avg_tuning_metrics(tuning)
    avg_dsis, avg_thetas = avg_raw_tuning(dirs, thresh_areas)
    traces = get_traces(all_data)
    small_keys = ["0.0", "2.8125", "5.625", "8.4375"]
    big_keys = ["0.0", "45.0", "90.0", "180.0"]

    basic_ds = plot_basic_ds_metrics(theta_diffs, avg_dsis, np.abs(avg_thetas))
    raw_bells_small = plot_raw_metric(dirs_180, thresh_areas, small_keys, show_trials=False)
    raw_bells_big = plot_raw_metric(dirs_180, thresh_areas, big_keys, show_trials=False)
    fancy = plot_fancy_scatter_ds_metrics(theta_diffs, avg_dsis, np.abs(avg_thetas))
    dir_traces_small = plot_traces(dirs_180, traces, diff_keys=small_keys)
    dir_traces_big = plot_traces(dirs_180, traces, diff_keys=big_keys)
    peaks = plot_peak_responses(thresh_areas)

    if 1:
        basic_ds.savefig(fig_path + "basic_ds.png", bbox_inches="tight")
        raw_bells_small.savefig(fig_path + "raw_bells_small.png", bbox_inches="tight")
        raw_bells_big.savefig(fig_path + "raw_bells_big.png", bbox_inches="tight")
        fancy.savefig(fig_path + "fancy_ds.png", bbox_inches="tight")
        dir_traces_small.savefig(fig_path + "dir_traces_small.png", bbox_inches="tight")
        dir_traces_big.savefig(fig_path + "dir_traces_big.png", bbox_inches="tight")
        peaks.savefig(fig_path + "directional_peaks.png", bbox_inches="tight")
    plt.show()
