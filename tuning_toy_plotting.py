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
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 10))

    axes[0].scatter(theta_diffs, dsis)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("DSi", fontsize=13)
    
    axes[1].scatter(theta_diffs, thetas)
    axes[1].set_ylabel("Preferred Theta", fontsize=13)
    axes[1].set_xlabel("E-I Angle Difference", fontsize=13)

    clean_axes(axes)
    return fig


def plot_fancy_scatter_ds_metrics(theta_diffs, dsis, thetas):
    fig, ax = plt.subplots(1)

    scatter = ax.scatter(thetas, dsis, c=theta_diffs, cmap="jet")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Preferred Theta", fontsize=13)
    ax.set_ylabel("DSi", fontsize=13)

    ax.set_title("E-I Theta Difference -> Colour")
    fig.colorbar(scatter, ax=ax, orientation="vertical", pad=.1)
    return fig


def plot_raw_metric(dirs, base_metric, diff_keys=["0.0", "90.0", "180.0"]):
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
    trial_tuning = {}
    for k, v in base_metric.items():
        dsis, thetas = calc_tuning(v, dirs, dir_ax=1)
        trial_tuning[k] = {"DSis": dsis, "thetas": thetas}

    return trial_tuning


def avg_tuning(tuning_dict):
    avg_dsis = [np.mean(v["DSis"]) for v in tuning_dict.values()]
    avg_thetas = [np.mean(v["thetas"]) for v in tuning_dict.values()]
    return avg_dsis, avg_thetas


"""
So, surprisingly with my initial setup, the DSi is pretty flat over tuning
differences. This might be due to a saturation case? Spike / non-linear response
might be too easy to generate so whether E is aligned with I or not, you are
are getting equally sharp tuning. E too strong, I too weak, Nav too dense? Have
to fiddle with these parameters and see.
"""

if __name__ == "__main__":
    basest = "/mnt/Data/NEURONoutput/tuning/"

    with h5.File(basest + "theta_diff_exp.h5", "r") as f:
        all_data = unpack_hdf(f)

    theta_diffs = [float(t) for t in all_data.keys()]
    dirs = json.loads(all_data["0.0"]["params"])["dir_labels"]
    dirs_180 = np.vectorize(scale_180_from_360)(dirs)

    thresh_areas = thresh_area_tuning(all_data)
    tuning = trial_by_trial_tuning(dirs, thresh_areas)
    avg_dsis, avg_thetas = avg_tuning(tuning)

    basic_ds = plot_basic_ds_metrics(theta_diffs, avg_dsis, avg_thetas)
    raw_bells = plot_raw_metric(dirs_180, thresh_areas)
    fancy = plot_fancy_scatter_ds_metrics(theta_diffs, avg_dsis, avg_thetas)
    plt.show()
