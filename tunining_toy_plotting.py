import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from experiments import Rig


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


def plot_basic_ds_metrics(base_metric):
    pass


def plot_fancy_scatter_ds_metrics(base_metric):
    pass


def plot_raw_metric(base_metric):
    pass


def plot_raw_metric(base_metric):
    pass


if __name__ == "__main__":
    basest = "/mnt/Data/NEURONoutput/tuning/"
