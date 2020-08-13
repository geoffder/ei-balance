import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import json

from analysis import calc_tuning
from general_utils import unpack_hdf, clean_axes


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


def plot_basic_ds_metrics(theta_diffs, dsis, thetas):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 10))

    axes[0].scatter(theta_diffs, dsis)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("DSi", fontsize=13)
    
    axes[1].scatter(theta_diffs, thetas)
    axes[1].set_ylabel("Preferred Theta", fontsize=13)
    axes[1].set_xlabel("E-I Angle Difference", fontsize=13)

    clean_axes(axes)
    plt.show()


def plot_fancy_scatter_ds_metrics(data):
    pass


def plot_raw_metric(data):
    pass


def trial_by_trial_tuning(dirs, data, metric_key):
    base = {k: d["metrics"][metric_key] for k, d in data.items()}

    trial_tuning = {}
    for k, v in base.items():
        dsis, thetas = calc_tuning(v, dirs, dir_ax=1)
        trial_tuning[k] = {"DSis": dsis, "thetas": thetas}

    return trial_tuning


def avg_tuning(tuning_dict):
    avg_dsis = [np.mean(v["DSis"]) for v in tuning_dict.values()]
    avg_thetas = [np.mean(v["thetas"]) for v in tuning_dict.values()]
    return avg_dsis, avg_thetas


if __name__ == "__main__":
    basest = "/mnt/Data/NEURONoutput/tuning/"

    with h5.File(basest + "theta_diff_exp.h5", "r") as f:
        all_data = unpack_hdf(f)

    theta_diffs = [float(t) for t in all_data.keys()]
    dirs = json.loads(all_data["0.0"]["params"])["dir_labels"]
    tuning = trial_by_trial_tuning(dirs, all_data, "peaks")
    avg_dsis, avg_thetas = avg_tuning(tuning)

    plot_basic_ds_metrics(theta_diffs, avg_dsis, avg_thetas)
