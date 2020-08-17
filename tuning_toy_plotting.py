import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import json

from analysis import calc_tuning
from general_utils import unpack_hdf, clean_axes
from modelUtils import scale_180_from_360


def thresholded_area(vm, thresh):
    """Take Vm ndarray (with time as last (or only) axis) and return
    with time dimension reduced to the area above the given threshold."""
    orig_shape = vm.shape
    reshaped = vm.reshape(-1, orig_shape[-1])
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
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 10))

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


def plot_raw_metric(dirs, base_metric, diff_keys=["0.0", "90.0", "180.0"]):
    """Plot response amplitude against stimulus direction for the given list
    of theta difference conditions."""
    fig, axes = plt.subplots(1, len(diff_keys), sharey=True, figsize=(12, 7))

    for ax, diff in zip(axes, diff_keys):
        ax.scatter(dirs, np.mean(base_metric[diff], axis=0))
        ax.set_xlabel("Direction", fontsize=13)
        ax.set_title("E-I Diff = %s" % diff)

    axes[0].set_ylabel("Response", fontsize=13)

    clean_axes(axes)
    fig.tight_layout()
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
    """Calculate average tunign by averaging tuning metrics (dsi and theta) for
    individual trials. Input dict has theta difference keys and values in the
    form of {"DSis": ndarray (1D), "thetas": ndarray (1D)}."""
    avg_dsis = [np.mean(v["DSis"]) for v in tuning_dict.values()]
    avg_thetas = [np.mean(v["thetas"]) for v in tuning_dict.values()]
    return np.concatenate(avg_dsis), np.concatenate(avg_thetas)


if __name__ == "__main__":
    basest = "/mnt/Data/NEURONoutput/tuning/"

    with h5.File(basest + "theta_diff_exp.h5", "r") as f:
        all_data = unpack_hdf(f)

    theta_diffs = np.array([float(t) for t in all_data.keys()])
    dirs = json.loads(all_data["0.0"]["params"])["dir_labels"]
    dirs_180 = np.vectorize(scale_180_from_360)(dirs)

    thresh_areas = thresh_area_tuning(all_data)
    tuning = trial_by_trial_tuning(dirs, thresh_areas)
    # avg_dsis, avg_thetas = avg_tuning_metrics(tuning)
    avg_dsis, avg_thetas = avg_raw_tuning(dirs, thresh_areas)
    
    basic_ds = plot_basic_ds_metrics(theta_diffs, avg_dsis, np.abs(avg_thetas))
    raw_bells = plot_raw_metric(dirs_180, thresh_areas)
    fancy = plot_fancy_scatter_ds_metrics(theta_diffs, avg_dsis, np.abs(avg_thetas))
    plt.show()
