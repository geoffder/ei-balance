# science/math libraries
from typing import Tuple, Optional
import numpy as np
import bottleneck as bn
from scipy import signal
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.figure import FigureBase
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
from skimage import io

# general libraries
import os  # make folders for file output
import h5py as h5
import json

# local libraries
from modelUtils import rotate, scale_180_from_360, wrap_180
from hdf_utils import unpack_hdf
from general_utils import clean_axes
from ei_balance import Model


def get_dsgc_img():
    img = np.flip(io.imread("assets/dsgc.png"), axis=0)
    y_px, x_px, _ = img.shape
    px_per_um = 2.55
    x_off = 2 / px_per_um
    y_off = 43 / px_per_um
    extent = (x_off, x_off + x_px / px_per_um, y_off + y_px / px_per_um, y_off)
    return img, extent


def nearest_index(arr, v):
    """Index of value closest to v in ndarray `arr`"""
    return bn.nanargmin(np.abs(arr - v))


def spike_transform(vm, kernel_sz, kernel_var, thresh=0):
    """Trying out a dead simple gaussian kernel transform to apply to Vm.
    Applies convolution to last dimension, all other dimensions should just
    be trials/locations/directions etc (e.g. data is just 1D time-series).
    Soma Vm is stored in shape (Trials, Directions, Time).
    """
    kernel = signal.gaussian(kernel_sz, kernel_var)
    og_shape = vm.shape
    vm = vm[:].reshape(-1, vm.shape[-1])

    spks, conv = np.zeros_like(vm), np.zeros_like(vm)

    for i, rec in enumerate(vm):
        idxs, _ = signal.find_peaks(rec, height=thresh)
        spks[i][idxs] = 1
        conv[i] = np.convolve(spks[i], kernel, mode="same")

    return conv.reshape(og_shape)


def gauss_conv(vm, kernel_sz, kernel_var):
    """Gaussian convolution of input recordings (without spike detection step)."""
    kernel = signal.gaussian(kernel_sz, kernel_var)
    og_shape = vm.shape
    vm = vm.reshape(-1, vm.shape[-1])
    conv = np.array([np.convolve(rec, kernel, mode="same") for rec in vm])
    return conv.reshape(og_shape)


def calc_tuning(data, dirs, dir_ax=0):
    """Calculate direction selective metrics of vector-sum index and preferred
    theta for the provided data. Shape of input array is flexible, though the
    directional dimension must be specified with dir_ax. Metrics are returned
    in the same shape as the input, minus the directional dimension."""
    data = np.moveaxis(data, dir_ax, 0)
    out_shape = data.shape[1:]

    data = data.reshape(data.shape[0], -1)
    rads = np.radians(dirs).reshape(-1, 1)

    xsums = bn.nansum(np.multiply(data, np.cos(rads)), axis=0)
    ysums = bn.nansum(np.multiply(data, np.sin(rads)), axis=0)
    DSi = np.squeeze(
        np.sqrt(xsums**2 + ysums**2) / (bn.nansum(data, axis=0) + 0.000001)
    )
    theta = np.squeeze(np.arctan2(ysums, xsums) * 180 / np.pi)

    return DSi.reshape(out_shape), theta.reshape(out_shape)


def generate_bar_sweep(stim):
    """Given a dict describing a drifting edge stimulus, generate two (T, 2)
    dimentional arrays describing the end coordinates of the edge at each
    time-step. e.g. (x1, x2) and (y1, y2) for each step t."""
    theta = np.radians(stim["theta"])
    ox, oy = rotate(stim["net_origin"], *stim["start"], theta)

    x_ends = np.array(
        [ox + np.cos(theta - np.pi / 2) * 200, ox + np.cos(theta + np.pi / 2) * 200]
    )
    y_ends = np.array(
        [oy + np.sin(theta - np.pi / 2) * 200, oy + np.sin(theta + np.pi / 2) * 200]
    )

    x_step = stim["dt"] * stim["speed"] * np.cos(theta)
    y_step = stim["dt"] * stim["speed"] * np.sin(theta)
    bar_x = np.array([x_ends + x_step * t for t in range(0, stim["n_pts"])])
    bar_y = np.array([y_ends + y_step * t for t in range(0, stim["n_pts"])])

    return bar_x, bar_y


def activity_gif(
    data,
    locs,
    fname,
    vmin=-63,
    vmax=10,
    sample_inter=25,
    rec_dt=0.1,
    gif_inter=150,
    dpi=100,
):
    """Create animated gif depicting dendritic activity in response to a
    simulated visual stimulus. A bar depicting the edge of the stimulus is
    overlaid, and the somatic Vm recording is included underneath."""
    fig, ax = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [0.8, 0.2]}, figsize=(5, 6)
    )

    scat = ax[0].scatter(locs[0], locs[1], c=data["tree"][:, 0], vmin=vmin, vmax=vmax)

    bar_x, bar_y = generate_bar_sweep(data["stim"])
    stim = ax[0].plot(bar_x[0], bar_y[0], c="black", linewidth=3)
    ax[0].set_xlim(-30, 230)
    ax[0].set_ylim(-30, 260)

    time = np.linspace(0, len(data["soma"]) * rec_dt, len(data["soma"]))
    line = ax[1].plot(time[0], data["soma"][0])
    ax[1].set_xlim(time.min(), time.max())
    ax[1].set_ylim(-70, 30)

    rate_diff = data["soma"].shape[0] // data["tree"].shape[1]

    def update(t):
        scat.set_array(data["tree"][:, t])
        line[0].set_xdata(time[: t * rate_diff])
        line[0].set_ydata(data["soma"][: t * rate_diff])
        stim[0].set_xdata(bar_x[t * rate_diff])
        stim[0].set_ydata(bar_y[t * rate_diff])
        return scat, line, stim

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, data["tree"].shape[1], sample_inter),
        interval=gif_inter,
    )
    anim.save(fname, dpi=dpi, writer="imagemagick")
    plt.close()


def make_sac_rec_gifs(
    pth,
    data,
    dirs,
    rec_key="Vm",
    rhos=[],
    net=0,
    trial=0,
    sample_inter=25,
    gif_inter=150,
    dpi=100,
):
    """For a given DSGC net and trial in the data dict, create + save gifs for
    each stimulus direction."""
    gif_pth = pth + "gifs/"
    if not os.path.isdir(gif_pth):
        os.mkdir(gif_pth)

    rhos = list(data.keys()) if rhos == [] else rhos
    for rho in rhos:
        if rec_key == "trans_Vm":
            data[rho][net]["dendrites"]["trans_Vm"] = spike_transform(
                data[rho][net]["dendrites"]["Vm"], 500, 30
            )
        rec_locs = data[rho][net]["dendrites"]["locs"]
        recs = data[rho][net]["dendrites"][rec_key]
        soma_recs = data[rho][net]["soma"]["Vm"]
        ps = data[rho][net]["params"]
        vmin, vmax = np.min(recs[trial]), np.max(recs[trial])

        for d, soma, dir_recs in zip(dirs, soma_recs[trial], recs[trial]):
            bar_start = np.array([ps["light_bar"]["x_start"], ps["origin"][1]])
            gif_data = {
                "soma": soma,
                "tree": dir_recs,
                "stim": {
                    "theta": d,
                    "net_origin": ps["origin"],
                    "start": bar_start,
                    "speed": ps["light_bar"]["speed"],
                    "dt": ps["dt"],
                    "n_pts": soma.shape[0],
                },
            }
            activity_gif(
                gif_data,
                rec_locs,
                gif_pth + "rho%s_tr%i_dir%i_%s.gif" % (rho, trial, d, rec_key),
                vmin=vmin,
                vmax=vmax,
                sample_inter=sample_inter,
                gif_inter=gif_inter,
                dpi=dpi,
            )


def tuning_gif(
    data, locs, fname, dsi_mul=250, sample_inter=25, rec_dt=0.1, gif_inter=150, dpi=100
):
    """Create animated gif depicting directional preference in response to a
    simulated visual stimulus."""
    data["tree"]["DSi"] *= dsi_mul

    fig, ax = plt.subplots(
        3, 1, gridspec_kw={"height_ratios": [0.7, 0.15, 0.15]}, figsize=(6, 7)
    )
    ax[1].set_ylabel("theta (°)", size=14)
    ax[2].set_ylabel("DSi", size=14)
    ax[1].set_xticklabels([])
    ax[2].set_xlabel("Time", size=14)

    scat = ax[0].scatter(
        locs[0],
        locs[1],
        c=data["tree"]["theta"][:, 0],
        s=data["tree"]["DSi"][:, 0],
        vmin=data["tree"]["min_theta"],
        vmax=data["tree"]["max_theta"],
    )
    time = np.linspace(0, len(data["soma"]["DSi"]) * rec_dt, len(data["soma"]["DSi"]))
    theta_line = ax[1].plot(time[0], data["soma"]["theta"][0])
    DSi_line = ax[2].plot(time[0], data["soma"]["DSi"][0])

    ax[1].set_xlim(time.min(), time.max())
    ax[2].set_xlim(time.min(), time.max())
    ax[1].set_ylim(
        data["soma"]["theta"].min(),
        data["soma"]["theta"].max(),
    )
    ax[2].set_ylim(
        data["soma"]["DSi"].min(),
        data["soma"]["DSi"].max(),
    )

    rate_diff = data["soma"]["DSi"].shape[0] // data["tree"]["DSi"].shape[1]

    # draw legend mapping DSi -> size with min/max examples.
    mn = ax[0].scatter(
        [], [], c="black", s=data["tree"]["min_DSi"] * dsi_mul, edgecolors="none"
    )
    mx = ax[0].scatter(
        [], [], c="black", s=data["tree"]["max_DSi"] * dsi_mul, edgecolors="none"
    )
    labels = ["%.2f" % data["tree"]["%s_DSi" % m] for m in ["min", "max"]]
    legend = fig.legend(
        [mn, mx],
        labels,
        ncol=2,
        frameon=True,
        fontsize=12,
        handlelength=2,
        loc="upper right",
        borderpad=1,
        handletextpad=1,
        title="DSi Bounds",
        title_fontsize=18,
    )
    fig.colorbar(scat, ax=ax[0], orientation="vertical", pad=0.1)

    def update(t):
        scat.set_array(data["tree"]["theta"][:, t])
        scat.set_sizes(data["tree"]["DSi"][:, t])
        theta_line[0].set_xdata(time[: t * rate_diff])
        theta_line[0].set_ydata(data["soma"]["theta"][: t * rate_diff])
        DSi_line[0].set_xdata(time[: t * rate_diff])
        DSi_line[0].set_ydata(data["soma"]["DSi"][: t * rate_diff])
        return scat, theta_line, DSi_line

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, data["tree"]["DSi"].shape[1], sample_inter),
        interval=gif_inter,
    )
    anim.save(fname, dpi=dpi, writer="imagemagick")
    plt.close()


def make_sac_tuning_gifs(
    pth,
    data,
    dirs,
    rec_key="Vm",
    rhos=[],
    net=0,
    thresh=None,
    sample_inter=25,
    gif_inter=150,
    dpi=100,
):
    """For a given DSGC net, create a gif for each trial of DSi/theta evolving
    over time for the tree, and soma (DSi/theta in same line plot).
    """
    gif_pth = pth + "gifs/"
    if not os.path.isdir(gif_pth):
        os.mkdir(gif_pth)

    rhos = list(data.keys()) if rhos == [] else rhos
    for rho in rhos:
        rec_locs = data[rho][net]["dendrites"]["locs"]

        soma_recs = data[rho][net]["soma"]["Vm"]
        if rec_key == "spk_Vm" or rec_key == "gauss_Vm":
            soma_recs = spike_transform(soma_recs, 500, 30)
            # soma_recs = gauss_conv(soma_recs, 500, 30)

        if rec_key == "spk_Vm":
            recs = spike_transform(
                data[rho][net]["dendrites"]["Vm"], 500, 30, thresh=-20
            )
        elif rec_key == "gauss_Vm":
            recs = gauss_conv(data[rho][net]["dendrites"]["Vm"], 500, 15)
        else:
            recs = data[rho][net]["dendrites"][rec_key]

        if thresh is not None:
            recs = np.clip(recs, thresh, None) - thresh
            soma_recs = np.clip(soma_recs, thresh, None) - thresh
        else:
            recs = recs - recs.min()
            soma_recs = soma_recs - soma_recs.min()

        for t in range(recs.shape[0]):
            # direction selective tuning for each point in time
            tree_DSi, tree_theta = calc_tuning(recs[t], dirs, dir_ax=0)
            soma_DSi, soma_theta = calc_tuning(soma_recs[t], dirs, dir_ax=0)

            tuning = {
                "soma": {"DSi": soma_DSi, "theta": soma_theta},
                "tree": {"DSi": tree_DSi, "theta": tree_theta},
            }

            # get theta and DSi ranges for colour and size scaling
            for k in tuning.keys():
                for m in list(tuning[k].keys()):
                    tuning[k]["min_%s" % m] = np.abs(tuning[k][m]).min()
                    tuning[k]["max_%s" % m] = np.abs(tuning[k][m]).max()

            tuning_gif(
                tuning,
                rec_locs,
                gif_pth + "rho%s_tr%i_tuning_%s.gif" % (rho, t, rec_key),
                sample_inter=sample_inter,
                gif_inter=gif_inter,
                dpi=dpi,
            )


def polar_plot(
    metrics,
    dirs,
    radius=None,
    net_shadows=False,
    title=None,
    title_metrics=True,
    save=False,
    save_pth="",
    save_ext="png",
    fig: Optional[FigureBase] = None,
    sub_loc=(1, 1, 1),
    dsi_tag_deg=-15.0,
    dsi_tag_mult=1.05,
    dsi_tag_fmt="DSi\n%.2f",
):
    # re-sort directions and make circular for polar axes
    circ_vals = metrics["spikes"].transpose(2, 0, 1)[np.array(dirs).argsort()]
    circ_vals = np.concatenate(
        [circ_vals, np.expand_dims(circ_vals[0], axis=0)], axis=0
    )
    circle = np.radians([0, 45, 90, 135, 180, 225, 270, 315, 0])

    peak = np.max(circ_vals)  # to set axis max
    thetas = np.radians(metrics["thetas"])
    DSis = metrics["DSis"]
    avg_DSi, avg_theta = calc_tuning(
        metrics["spikes"].reshape(-1, len(dirs)).sum(axis=0), dirs
    )
    avg_theta = np.radians(avg_theta)

    if fig is None:
        fig = plt.figure(figsize=(5, 6))
        new_fig = True
    else:
        new_fig = False

    ax = fig.add_subplot(*sub_loc, projection="polar")

    # plot trials lighter
    if net_shadows:
        shadows = np.mean(circ_vals, axis=2)
        thetas = np.radians(metrics["avg_theta"])
        DSis = metrics["avg_DSi"]
    else:
        shadows = circ_vals.reshape(circle.size, -1)
        thetas = thetas.flatten()
        DSis = DSis.flatten()
    ax.plot(circle, shadows, color=".75")
    ax.plot([thetas, thetas], [np.zeros_like(DSis), DSis * peak], color=".75")

    # plot avg darker
    avg_spikes = np.mean(circ_vals.reshape(circle.size, -1), axis=1)
    ax.plot(circle, avg_spikes, color=".0", linewidth=2)
    ax.plot([avg_theta, avg_theta], [0.0, avg_DSi * peak], color=".0", linewidth=2)

    # misc settings
    radius = peak if radius is None else radius
    ax.set_rmax(radius)
    ax.set_rticks([radius])
    ax.set_rlabel_position(45)
    ax.set_thetagrids([0, 90, 180, 270])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=15)
    if title_metrics:
        sub = "DSi = %.2f :: θ = %.2f\n  σ = %.2f :: σ = %.2f" % (
            avg_DSi,
            np.degrees(avg_theta),
            np.std(DSis),
            np.std(np.vectorize(scale_180_from_360)(np.degrees(thetas - avg_theta))),
        )
        title = sub if title is None else "%s\n%s" % (title, sub)
    else:
        ax.text(
            np.radians(dsi_tag_deg),
            radius * dsi_tag_mult,
            dsi_tag_fmt % avg_DSi,
            fontsize=13,
        )

    if title is not None:
        ax.set_title(title, size=15)

    if save:
        fig.savefig(
            "%spolar_%s.%s" % (save_pth, title.replace(" ", "_"), save_ext),
            bbox_inches="tight",
        )

    if new_fig:
        return fig, ax
    else:
        return ax


def load_sac_rho_data(pth, prefix="sac_rho"):
    """Collect data from each of the hdf5 archives located in the target folder
    representing rho conditions and their trials and store them in a dict."""
    all_exps = [
        f
        for f in os.listdir(pth)
        if prefix in f and os.path.isfile(os.path.join(pth, f))
    ]

    # assumes rho is .2f precision following prefix. Should make less
    # hacky at some point.
    fnames = {
        r: [f for f in all_exps if r in f]
        for r in set([n[len(prefix) : len(prefix) + 4] for n in all_exps])
    }

    sac_data = {}
    for r, names in fnames.items():
        sac_data[r] = {}
        for i, n in enumerate(names):
            with h5.File(pth + n, "r") as f:
                sac_data[r][i] = unpack_hdf(f)
            sac_data[r][i]["params"] = json.loads(sac_data[r][i]["params"])

    return sac_data


def load_dir_vc_data(pth):
    with h5.File(pth, "r") as f:
        vc_data = unpack_hdf(f)
    vc_data["params"] = json.loads(vc_data["params"])

    return vc_data


def get_sac_metrics(data):
    """Collect the already calculated metrics (trial spike numbers, preferred
    thetas and DSis) into a simpler dictionary."""
    metrics = {
        r: {
            m: np.stack([e["metrics"][m] for e in exps.values()], axis=0)
            for m in ["spikes", "thetas", "DSis", "avg_theta", "avg_DSi"]
        }
        for r, exps in data.items()
    }
    return metrics


def sac_rho_polars(
    metrics, dirs, save=False, net_shadows=False, save_pth="", save_ext="png"
):
    """Collect pre-calculated metrics from given data dict and feed them in to
    polar_plot function for each sac angle rho level included in the data."""
    max_spikes = np.max([r["spikes"] for r in metrics.values()])

    polar_figs = [
        polar_plot(
            m,
            dirs,
            title="rho " + str(r),
            radius=max_spikes,
            net_shadows=net_shadows,
            save=save,
            save_pth=save_pth,
            save_ext=save_ext,
        )
        for r, m in metrics.items()
    ]

    return polar_figs


def sac_rho_violins(metrics, dir_labels, viol_inner="box", **plot_kwargs):
    fig, ax = plt.subplots(4, 1, sharex=True, **plot_kwargs)

    labels = [k for k, v in metrics.items() for _ in range(v["DSis"].size)]
    dsis = np.concatenate([r["DSis"].flatten() for r in metrics.values()])
    sns.violinplot(x=labels, y=dsis, inner=viol_inner, ax=ax[0])
    ax[0].set_ylabel("DSi", size=14)
    ax[0].set_ylim(0)

    # for each rho level, reshape to (Directions, Trials * Nets)
    dir_spks = {
        k: v["spikes"].transpose().reshape(v["spikes"].shape[-1], -1)
        for k, v in metrics.items()
    }
    idxs = [np.argwhere(dir_labels == d)[0][0] for d in [135, 180, 225]]

    for a, idx in zip(ax[1:], idxs):
        spks = np.concatenate([r[idx] for r in dir_spks.values()])
        max_spks = np.max(spks)
        max_tick = max_spks if max_spks % 2 == 0 else max_spks + 1
        sns.violinplot(x=labels, y=spks, inner=viol_inner, ax=a)
        a.set_ylabel("Spikes in %d" % dir_labels[idx], size=14)
        a.set_ylim(-1)
        a.set_yticks([0, max_tick // 2, max_tick])

    ax[-1].set_xlabel("Correlation (rho)", size=14)
    fig.tight_layout()

    return fig


def dendritic_ds(tree_recs, dirs, pref=0, thresh=None, bin_pts=None):
    """Expected dendrite recordings are in an array of shape (N, Dirs, Locs, T).
    If passing in an average, include a singleton 0th dimension.
    """
    if bin_pts is not None:
        tree_recs = tree_recs[:, :, :, bin_pts[0] : bin_pts[1]]

    if thresh is not None:
        tree_recs = np.clip(tree_recs, thresh, None) - thresh
    else:
        tree_recs = tree_recs - bn.nanmin(tree_recs)

    areas = bn.nansum(tree_recs, axis=3)

    # direction selective tuning for each recording location on tree
    DSis, thetas = calc_tuning(areas, dirs, dir_ax=1)
    if pref != 0:
        # NOTE: This is on 0 -> 180 scale, so ABSOLUTE theta diff from preferred
        # NOT -180 -> 180 scale. Make sure that this is what I want.
        thetas = np.vectorize(scale_180_from_360)(thetas - pref)

    return {"DSi": DSis, "theta": thetas}


def analyze_tree(data, dirs, rec_key="Vm", pref=0, thresh=None, bin_pts=None):
    """Build dendritic tree tuning (preferred thetas and DSis for each
    recording location) dict containing all of the DSGCs at each rho level
    included in the given data dict."""
    tuning = {}
    for cond, nets in data.items():
        tuning[cond] = {}
        for i in nets.keys():
            tree_recs = data[cond][i]["dendrites"][rec_key]
            avg_recs = np.expand_dims(bn.nanmean(tree_recs, axis=0), 0)
            tuning[cond][i] = {
                "trials": dendritic_ds(
                    tree_recs, dirs, pref=pref, thresh=thresh, bin_pts=bin_pts
                ),
                "avg": dendritic_ds(
                    avg_recs, dirs, pref=pref, thresh=thresh, bin_pts=bin_pts
                ),
                "locs": data[cond][i]["dendrites"]["locs"],
            }

    return tuning


def plot_tree_tuning(tuning_dict, net_idx, trial=None, dsi_size=True, dsi_mul=250):
    """Plot theta/DSi tuning of dendrites for a given (by index) network, for
    each sac angle rho in the given tuning_dict."""
    if trial is None:
        trial = 0
        level = "avg"
    else:
        level = "trials"

    fig, axes = plt.subplots(1, len(tuning_dict), figsize=(12, 7))

    # range vars for size and colour legend
    min_dsi, max_dsi = 1.0, 0.0
    min_theta, max_theta = 180.0, 0.0

    # get theta range for colour scaling
    for rho, nets in tuning_dict.items():
        net = nets[net_idx]
        rel_theta = np.abs(net[level]["theta"][trial])

        net_low = rel_theta.min()
        min_theta = net_low if net_low < min_theta else min_theta
        net_high = rel_theta.max()
        max_theta = net_high if net_high > max_theta else max_theta

        net_low = net[level]["DSi"][trial].min()
        min_dsi = net_low if net_low < min_dsi else min_dsi
        net_high = net[level]["DSi"][trial].max()
        max_dsi = net_high if net_high > max_dsi else max_dsi

    # sort dict so rho increases left to right
    for (rho, nets), ax in zip(sorted(tuning_dict.items()), axes):
        net = nets[net_idx]

        sz = net[level]["DSi"][trial] * dsi_mul if dsi_size else 20
        scatter = ax.scatter(
            net["locs"][0],
            net["locs"][1],
            s=sz,
            alpha=0.5,
            c=np.abs(net[level]["theta"][trial]),
            cmap="jet",
            vmin=min_theta,
            vmax=max_theta,
        )
        ax.set_title("rho = %s" % rho, fontsize=18)

    # draw legend mapping DSi -> size with min/max examples.
    if dsi_size:
        mn = axes[0].scatter([], [], c="black", s=min_dsi * dsi_mul, edgecolors="none")
        mx = axes[0].scatter([], [], c="black", s=max_dsi * dsi_mul, edgecolors="none")
        labels = ["%.2f" % m for m in [min_dsi, max_dsi]]
        legend = fig.legend(
            [mn, mx],
            labels,
            ncol=2,
            frameon=True,
            fontsize=12,
            handlelength=2,
            loc="upper right",
            borderpad=1,
            handletextpad=1,
            title="DSi Bounds",
            title_fontsize=18,
        )

    fig.colorbar(scatter, ax=axes, orientation="horizontal", pad=0.1)

    return fig


def ds_scatter(tuning_dict, x_max=None, **plot_kwargs):
    fig, axes = plt.subplots(1, len(tuning_dict), sharey=True, **plot_kwargs)
    axes = axes if len(tuning_dict) > 1 else [axes]
    dsi_max = 0.0

    # sort dict so rho increases left to right
    for (cond, nets), ax in zip(sorted(tuning_dict.items()), axes):
        pts = {
            m: np.concatenate([n["trials"][m].flatten() for n in nets.values()])
            for m in ["DSi", "theta"]
        }
        dsi_max = max(dsi_max, np.max(pts["DSi"]))
        ax.scatter(pts["DSi"], pts["theta"], c="black", alpha=0.1)
        ax.set_xlabel("DSi", size=14)
        title = cond if type(cond) is str else "rho = %s" % cond
        ax.set_title(title, fontsize=18)

    axes[0].set_ylim(-180, 180)
    axes[0].set_ylabel("theta (°)", size=14)

    for ax in axes:
        if x_max is None:
            ax.set_xlim(0, dsi_max)
        else:
            ax.set_xlim(0, x_max)

    return fig


def spike_rasters(
    data,
    dirs,
    net_idx=None,
    rho=None,
    thresh=0,
    bin_ms=0,
    offsets=None,
    colour="black",
    spike_vmax=None,
    **plot_kwargs,
):
    """Plot spike-time raster for each direction, at each rho level. Spikes are
    pooled across trials, and also networks unless a net_idx is specified.
    Optionally, display binned tuning calculations beneath raster."""
    rhos = list(data.keys())
    dt = data[rhos[0]][0]["params"]["dt"]
    rec_sz = data[rhos[0]][0]["soma"]["Vm"].shape[-1]
    rel_dirs = [d if d <= 180 else d - 360 for d in dirs]

    data = data if rho is None else {rho: data[rho]}
    if bin_ms < 1:
        fig, axes = plt.subplots(
            1,
            len(data),
            sharey="row",
            squeeze=False,
            **plot_kwargs,
        )
    else:
        fig, axes = plt.subplots(
            4,
            len(data),
            sharex="col",
            sharey="row",
            gridspec_kw={"height_ratios": [0.6, 0.133, 0.133, 0.133]},
            **plot_kwargs,
        )
        pts_per_bin = int(bin_ms / dt)
        bin_pts = np.array(
            [pts_per_bin * i for i in range(1, rec_sz // pts_per_bin + 1)]
        )
        bin_centres = dt * (bin_pts - pts_per_bin / 2)  # for plotting

    # unzip axes from rows in to column-major organization
    if rho is None:
        axes = [[a[i] for a in axes] for i in range(len(data))]
    else:
        axes = [axes]

    # sort dict so rho increases left to right
    for (rho, nets), col in zip(sorted(data.items()), axes):
        dir_ts = {d: [] for d in dirs}
        for i, net in nets.items():
            if net_idx is not None and net_idx != i:
                continue
            for trial in net["soma"]["Vm"]:
                for d, rec in zip(dirs, trial):
                    idxs, _ = signal.find_peaks(rec, height=thresh)
                    dir_ts[d].append(idxs * dt)

        dir_ts = {d: np.concatenate(ts) for d, ts in dir_ts.items()}
        if offsets is not None:
            dir_ts = {d: ts + off for (d, ts), off in zip(dir_ts.items(), offsets)}

        col[0].eventplot(
            dir_ts.values(),
            lineoffsets=rel_dirs,
            linelengths=30,
            colors=colour,
            alpha=0.5,
        )

        if bin_ms > 0:
            cum_spks = [np.zeros(len(dir_ts))]
            for p in bin_pts:
                # mark bin margins on raster plot
                col[0].axvline(
                    p * net["params"]["dt"], c="black", linestyle="--", alpha=0.5
                )
                # number of spikes before each bin margin (for each direction)
                cum_spks.append([np.sum(ts < (p * dt)) for ts in dir_ts.values()])
            cum_spks = np.array(cum_spks)
            bin_spks = (cum_spks[1:] - cum_spks[0:-1]).T

            # calculate and plot theta/DSi for each bin
            mean_DSi, mean_theta = calc_tuning(bin_spks, dirs, dir_ax=0)

            style = {"color": colour, "linestyle": "--", "marker": "D"}
            col[1].plot(bin_centres, np.sum(bin_spks, axis=0), **style)
            col[1].set_ylim(0, spike_vmax)
            col[2].plot(bin_centres, mean_theta, **style)
            col[3].plot(bin_centres, mean_DSi, **style)

        # shared X settings
        col[0].set_xlim(0, rec_sz * dt)
        col[0].set_title("rho = %s" % rho, fontsize=18)
        col[-1].set_xlabel("Time (ms)", size=14)

    # shared Y settings
    axes[0][0].set_ylabel("Direction (°)", size=14)
    axes[0][0].set_yticks(rel_dirs)
    axes[0][1].set_ylabel("Total Spikes", size=14)
    axes[0][2].set_ylabel("theta (°)", size=14)
    axes[0][2].set_ylim(-180, 180)
    axes[0][3].set_ylabel("DSi", size=14)
    axes[0][3].set_ylim(0, 1)

    fig.tight_layout()

    return fig


def time_evolution(data, dirs, kernel_var=30, rhos=None, net_idx=None, **plot_kwargs):
    """Apply a continuous transform to spike somatic spike train and use to
    calculate theta and DSi for each moment in time. Display spike waveforms
    (aggregate mean with per network means) for each direction, with theta and
    DSi below."""
    rhos = list(data.keys()) if rhos is None else rhos
    dt = data[rhos[0]][0]["params"]["dt"]
    rec_sz = data[rhos[0]][0]["soma"]["Vm"].shape[-1]
    time = np.linspace(0, rec_sz * dt, rec_sz)

    fig, axes = plt.subplots(10, len(rhos), sharex="col", sharey="row", **plot_kwargs)
    # unzip axes from rows in to column-major organization
    if len(rhos) > 1:
        axes = [[a[i] for a in axes] for i in range(len(rhos))]
    else:
        axes = [axes]

    # sort dict so rho increases left to right
    for rho, col in zip(rhos, axes):
        nets = data[rho]
        spk_waves = [
            spike_transform(nets[n]["soma"]["Vm"], 500, kernel_var)
            for n in nets.keys()
            if net_idx is None or net_idx == n
        ]
        net_means = np.stack([np.mean(w, axis=0) for w in spk_waves], axis=1)
        mean_waves = np.mean(np.concatenate(spk_waves, axis=0), axis=0)

        # plot average spike waves for each direction
        for ax, ns, wave in zip(col, net_means, mean_waves):
            for n in ns:
                ax.plot(time, n, c="tab:blue", linewidth=0.75, alpha=0.5)
            ax.plot(time, wave, c="tab:blue", linewidth=2)

        # direction selective tuning for each point in time
        mean_DSi, mean_theta = calc_tuning(mean_waves, dirs, dir_ax=0)
        net_DSis, net_thetas = calc_tuning(net_means, dirs, dir_ax=0)

        for ts in net_thetas:
            col[-2].plot(time, ts, c="tab:blue", linewidth=0.75, alpha=0.5)
        col[-2].plot(time, mean_theta, c="tab:blue", linewidth=2)
        col[-2].set_ylim(-180, 180)

        for ds in net_DSis:
            col[-1].plot(time, ds, c="tab:blue", linewidth=0.75, alpha=0.5)
        col[-1].plot(time, mean_DSi, c="tab:blue", linewidth=2)
        col[-1].set_ylim(0, 1)

        col[0].set_title("rho = %s" % rho, fontsize=18)

    ymax = net_means.max()
    for col in axes:
        for ax in col[:-2]:
            if ymax > 0:
                ax.set_ylim(0, ymax)
        col[-1].set_xlim(time.min(), time.max())
        col[-1].set_xlabel("Time (ms)", size=14)

    for d, ax in zip(dirs, axes[0]):
        ax.set_ylabel(
            "%i°" % d, rotation=0, labelpad=20, size=14, verticalalignment="center"
        )

    axes[0][-1].set_ylabel("DSi", size=14)
    axes[0][-2].set_ylabel("theta (°)", size=14)
    # fig.tight_layout()

    return fig


def receptive_field_shift(vc_data):
    """Calculate ms to shift by (additive) to line up responses from each
    direction, accounting for the asymmetrical receptive field of the model.
    approx => [ 7.9, 20.6, 47.3, 15.1,  0. , 11.6, 23.4, 12.2]
    """
    ampa = np.mean(vc_data["soma"]["AMPA"], axis=0)
    edges = np.array([np.where(a)[0][0] for a in ampa])
    onset = edges * vc_data["params"]["dt"]
    return np.max(onset) - onset


def get_sac_thetas(data):
    """Pick out the dendrite angles from the sac wiring conditions."""
    thetas = {
        r: {
            trans: np.stack(
                [e["sac_net"]["wiring"]["thetas"][trans] for e in exps.values()], axis=0
            )
            for trans in ["E", "I"]
        }
        for r, exps in data.items()
    }
    return thetas


def get_sac_deltas(sac_thetas):
    """Calculate E/I theta difference for each synapse for each SAC network.
    Compare these to DSi/preferred theta of each synaptic site to gain insight
    into the importance of SAC dendrite alignment.
    NOTE: Includes nans, where there is no inhibitory input."""
    wrap = np.vectorize(wrap_180)
    deltas = {r: wrap(thetas["E"] - thetas["I"]) for r, thetas in sac_thetas.items()}
    return deltas


def get_syn_rec_lookups(rec_locs, syn_locs):
    """VERY hacky way of getting which recordings correspond to which synapses.
    Doing this as a quick way of looking at how pre-synaptic (SAC) arrangement
    impacts the post-synapse. Right now since the # of synapses are fewer than
    the recs, and they cover a variable amount of the tree, there isn't anything
    being stored (or easily added in quickly) to line them up to dendrites index
    wise. Also the recording electrodes are placed independantly.

    Here we line the synapses with particular recordings by using the XY
    locations that have been recorded for them.
    """
    syn_to_rec = {}
    rec_to_syn = {}
    for i in range(syn_locs.shape[0]):
        rec_idx = np.argmin(
            np.abs((syn_locs[i, 0] - rec_locs[0]))
            + np.abs((syn_locs[i, 1] - rec_locs[1]))
        )
        syn_to_rec[i] = rec_idx
        rec_to_syn[rec_idx] = i
    return {"to_rec": syn_to_rec, "to_syn": rec_to_syn}


def get_postsyn_avg_tuning(tuning_dict, lookups):
    d = {}
    for r, exps in tuning_dict.items():
        thetas = []
        dsis = []
        for e in exps.values():
            ts = [e["avg"]["theta"][:, idx] for idx in lookups["to_rec"].values()]
            ds = [e["avg"]["DSi"][:, idx] for idx in lookups["to_rec"].values()]
            thetas.append(ts)
            dsis.append(ds)
        d[r] = {
            "DSi": np.squeeze(np.stack(dsis, axis=0)),
            "theta": np.squeeze(np.stack(thetas, axis=0)),
        }
    return d


def plot_theta_diff_tuning_scatters(post_syn_avg_tuning, sac_deltas, rhos=[0.00]):
    """TODO: 4 panels. Cols: rhos Rows (Y): theta, dsi, sharedx: delta"""
    fig, axes = plt.subplots(3, len(rhos), sharex="col", figsize=(6 * len(rhos), 10))
    if len(rhos) > 1:
        # unzip axes from rows in to column-major organization
        axes = [[a[i] for a in axes] for i in range(len(rhos))]
    else:
        axes = [axes]

    for r, col in zip(rhos, axes):
        flat_deltas = sac_deltas[r].reshape(-1)
        flat_dsis = post_syn_avg_tuning[r]["DSi"].reshape(-1)
        flat_thetas = post_syn_avg_tuning[r]["theta"].reshape(-1)
        flat_thetas_abs = np.abs(flat_thetas)
        col[0].scatter(flat_deltas, flat_thetas)
        col[1].scatter(flat_deltas, flat_thetas_abs)
        col[2].scatter(flat_deltas, flat_dsis)
        # SAC Delta vs DSi and absolute theta fits
        not_nan_idxs = ~np.isnan(flat_deltas)
        nanless_deltas = flat_deltas[not_nan_idxs]
        nanless_dsis = flat_dsis[not_nan_idxs]
        nanless_thetas = flat_thetas_abs[not_nan_idxs]
        xaxis = np.linspace(0, np.max(nanless_deltas), 100)
        m, x, r_val, p_val, std_err = linregress(nanless_deltas, nanless_dsis)
        dsi_fit_line = x + m * xaxis
        m, x, r_val, p_val, std_err = linregress(nanless_deltas, nanless_thetas)
        theta_fit_line = x + m * xaxis
        col[1].plot(xaxis, theta_fit_line, c="red")
        col[2].plot(xaxis, dsi_fit_line, c="red")

        # shared X settings
        col[0].set_title("rho = %s" % r)
        col[2].set_xlabel("SAC Theta Delta (°)")

    # shared Y settings
    axes[0][0].set_ylabel("theta (°)", size=14)
    axes[0][0].set_ylim(-180, 180)
    axes[0][1].set_ylabel("absolute theta (°)", size=14)
    axes[0][1].set_ylim(0, 180)
    axes[0][2].set_ylabel("DSi", size=14)
    axes[0][2].set_ylim(0, 1)

    fig.tight_layout()
    return fig


def plot_theta_diff_vs_abs_theta(
    post_syn_avg_tuning, sac_deltas, rhos=[0.00], bin_sz=10
):
    fig, axes = plt.subplots(1, len(rhos))
    if len(rhos) > 1:
        # unzip axes from rows in to column-major organization
        axes = [[a[i] for a in axes] for i in range(len(rhos))]
    else:
        axes = [axes]

    for r, col in zip(rhos, axes):
        flat_deltas = sac_deltas[r].reshape(-1)
        not_nan_idxs = ~np.isnan(flat_deltas)
        flat_deltas = flat_deltas[not_nan_idxs]
        flat_thetas = post_syn_avg_tuning[r]["theta"].reshape(-1)
        flat_thetas_abs = np.abs(flat_thetas)[not_nan_idxs]
        binned = []
        for i in range(180 // bin_sz):
            idxs = (flat_deltas > (bin_sz * i)) * (flat_deltas < (bin_sz * (i + 1)))
            n = np.sum(idxs)
            if n > 0:
                binned.append(np.sum(flat_thetas_abs[idxs]) / n)
            else:
                binned.append(0)

        # shared X settings
        col.plot([i * bin_sz for i in range(1, 180 // bin_sz + 1)], binned)
        col.set_title("rho = %s" % r)
        col.set_xlabel("SAC Theta Delta Bin (°)")

    # shared Y settings
    axes[0].set_ylabel("absolute theta (°)", size=14)
    axes[0].set_ylim(0, 180)

    fig.tight_layout()
    return fig


def rough_gaba_weight_scaling():
    """Rough comparison of DSis with correlated and uncorrelated SAC nets with
    a range of GABA synapse weights. DSis are taken from the figures generated
    for 3 trial / 3 network runs of the spiking model with synaptic weight of
    GABA scaled by the factors found in `multi`."""
    rho0 = [0.03, 0.05, 0.08, 0.16, 0.21, 0.25, 0.34]
    rho1 = [0.05, 0.08, 0.14, 0.27, 0.35, 0.37, 0.44]
    multi = [0.10, 0.15, 0.25, 0.50, 0.75, 1.0, 1.5]

    fig, ax = plt.subplots()
    ax.plot(multi, rho1, c="black", label="Correlated")
    ax.plot(multi, rho0, c="red", label="Uncorrelated")
    ax.set_xlabel("GABA Weight Scaling", fontsize=15)
    ax.set_ylabel("DSi", fontsize=18)
    clean_axes(ax, ticksize=15)
    fig.legend(frameon=False, fontsize=18)
    fig.tight_layout()
    fig.show()


def rough_gaba_coverage_scaling():
    """Rough comparison of DSis with correlated and uncorrelated SAC nets with
    a range of GABA synapse weights. DSis are taken from the figures generated
    for 3 trial / 3 network runs of the spiking model with synaptic weight of
    GABA scaled by the factors found in `multi`."""
    rho0 = [0.04, 0.07, 0.13, 0.16, 0.22, 0.25, 0.25, 0.24, 0.23, 0.17]
    rho1 = [0.04, 0.08, 0.16, 0.21, 0.30, 0.32, 0.37, 0.39, 0.40, 0.38]
    multi = [0.10, 0.15, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

    fig, ax = plt.subplots()
    ax.plot(multi, rho0, label="rho 0")
    ax.plot(multi, rho1, label="rho 1")
    ax.set_xlabel("GABA Coverage")
    ax.set_ylabel("DSi")
    plt.show()


def sac_angle_distribution(
    config, n_nets=100, rho=1.0, bins=[8, 12, 16], incl_yticks=False, **plot_kwargs
):
    """Plot SAC dendrite angle distribution histograms, aggregating over a
    number of generated networks to get a more accurate estimate."""
    model = Model(config)
    iThetas, eThetas = [], []
    total_gaba = 0

    for _ in range(n_nets):
        model.build_sac_net(rho)
        iThetas.append(model.sac_net.thetas["I"][model.sac_net.gaba_here])
        eThetas.append(model.sac_net.thetas["E"])
        total_gaba += np.sum(model.sac_net.gaba_here)

    iThetas = np.concatenate(iThetas)
    eThetas = np.concatenate(eThetas)
    print("Average GABA synapse count: %.2f" % (total_gaba / n_nets))

    fig, axes = plt.subplots(1, len(bins), **plot_kwargs)
    axes = axes if len(bins) > 1 else [axes]

    for j, numBins in enumerate(bins):
        binAx = [i * 360 / numBins for i in range(numBins)]
        lbl_e, lbl_i = ("ACh", "GABA") if not j else (None, None)
        axes[j].hist(eThetas, bins=binAx, color="g", alpha=0.5, label=lbl_e)
        axes[j].hist(iThetas, bins=binAx, color="m", alpha=0.5, label=lbl_i)
        if len(bins) > 1:
            axes[j].set_title("Bin Size: %.1f" % (360 / numBins))

        axes[j].set_xlabel("SAC Dendrite Angle", fontsize=12.0)
        axes[j].set_ylabel("Count", fontsize=12.0)
        if not incl_yticks:
            axes[j].set_yticks([])

    fig.legend(frameon=False, fontsize=14)
    clean_axes(axes, ticksize=12.0)

    return fig, axes


def plot_dends_overlay(
    fig,
    ax,
    syn_locs,
    bp_locs,
    probs,
    dirs,
    dsgc_alpha=0.35,
    sac_marker_size=120,
    sac_thickness=None,
    sac_alpha=0.7,
    stim_angle=None,
    n_syn=None,
    cmap="plasma",
    syn_choice_seed=None,
    show_plex=False,
):
    # flip y-axis of image to match it up with coordinate system
    dsgc_img, extent = get_dsgc_img()

    # offset whitespace and convert um locations to px
    syn_xs = syn_locs[:, 0]
    syn_ys = syn_locs[:, 1]
    ach_xs = bp_locs["E"][:, 0]
    ach_ys = bp_locs["E"][:, 1]
    gaba_xs = bp_locs["I"][:, 0]
    gaba_ys = bp_locs["I"][:, 1]

    ax.imshow(dsgc_img, extent=extent, alpha=dsgc_alpha, cmap="gray")
    ax.set_xlim(-30, 230)
    ax.set_ylim(-30, 260)

    def plot_stick(syn_x, syn_y, bp_x, bp_y, clr):
        x = [syn_x, bp_x]
        y = [syn_y, bp_y]
        ax.plot(x, y, c=clr, linewidth=sac_thickness, alpha=sac_alpha)

    rng = np.random.default_rng(syn_choice_seed)
    idxs = rng.choice(
        len(syn_xs), size=len(syn_xs) if n_syn is None else n_syn, replace=False
    )
    if stim_angle is not None:
        colors = [plt.get_cmap(cmap)(1.0 * i / 100) for i in range(100)]
        angle_idx = np.argwhere(dirs == np.array(stim_angle))[0][0]
    else:
        colors, angle_idx = [], 0

    for i in idxs:
        if show_plex and "PLEX" in bp_locs:
            for x, y in bp_locs["PLEX"][i]:
                plot_stick(syn_xs[i], syn_ys[i], x, y, "g")
        plot_stick(syn_xs[i], syn_ys[i], ach_xs[i], ach_ys[i], "g")
        if not np.isnan(gaba_xs[i]):  # type: ignore
            plot_stick(syn_xs[i], syn_ys[i], gaba_xs[i], gaba_ys[i], "m")
        scat_kwargs = {"s": sac_marker_size, "alpha": sac_alpha}

        if show_plex and "PLEX" in bp_locs:
            for (x, y), prob in zip(bp_locs["PLEX"][i], probs["PLEX"][i][angle_idx]):
                clr = "g" if stim_angle is None else [colors[int(prob / 0.01)]]
                ax.scatter(x, y, c=clr, **scat_kwargs)

        prob = probs["E"][i][angle_idx]
        clr = "g" if stim_angle is None else [colors[int(prob / 0.01)]]
        ax.scatter(ach_xs[i], ach_ys[i], c=clr, **scat_kwargs)

        if not np.isnan(gaba_xs[i]):  # type: ignore
            prob = probs["I"][i][angle_idx]
            clr = "m" if stim_angle is None else [colors[int(prob / 0.01)]]
            ax.scatter(gaba_xs[i], gaba_ys[i], marker="v", c=clr, **scat_kwargs)

    if stim_angle is not None:
        cbar = fig.colorbar(
            ScalarMappable(Normalize(vmin=0.0, vmax=1.0), cmap=cmap),
            ax=ax,
            orientation="horizontal",
            pad=0.0,
            shrink=0.75,
        )
        cbar.ax.tick_params(labelsize=12.0)
        cbar.set_label("Release Probability", fontsize=12)

    ach_circ = ax.scatter([], [], c="g", s=sac_marker_size)
    gaba_tri = ax.scatter([], [], c="m", marker="v", s=sac_marker_size)
    ax.legend(
        [ach_circ, gaba_tri],
        ["ACh", "GABA"],
        ncol=1,
        frameon=False,
        fontsize=12,
        handlelength=2,
        loc="upper right",
        bbox_to_anchor=(0.35, 0.25),
        borderpad=1,
        handletextpad=1,
        title_fontsize=12,
    )

    clean_axes(ax, remove_spines=["left", "right", "top", "bottom"])


# TODO: "Zoom-in" on synapses / recordings locations that have drammatically off
# post-synaptic tuning (mainly TTX of course), and log what the SAC angles were.
# Another way of trying to bring out the effect a bit more cleanly.

# TODO: Enable rasters, tuning evolution, and violins to adjust to different
# preferred directions as well.
if __name__ == "__main__":
    basest = "/mnt/Data/NEURONoutput/sac_net/"
    # basest += "uni_var60_E90_I90_ARM_nonDirGABA/"
    # basest += "ttx/"
    # basest += "gaba_titration/ttx/gaba_scale_150p/"
    # basest += "committee_runs/spiking/control/"
    # basest += "committee_runs/spiking/non_ds_ach_50pr/"
    basest += "test_sigmoid/spiking/non_ds_ach_50pr/"
    # basest += "spiking_cable_diam/"
    # basest += "ttx_cable_diam/"
    fig_pth = basest + "py_figs/"
    fig_exts = ["png", "svg"]

    def save_fig(f, n):
        for ext in fig_exts:
            f.savefig(fig_pth + n + "." + ext, bbox_inches="tight")

    if not os.path.isdir(fig_pth):
        os.mkdir(fig_pth)

    dir_labels = [225, 270, 315, 0, 45, 90, 135, 180]
    # NOTE: calculated from avg VC recordings of E current (for current settings)
    # dir_field_offsets = [9.1, 17.1, 26.2, 14.7, 2.0, 12.5, 11.0, 0.0]  # ms
    # dir_field_offsets = [9.6, 19.1, 36.2, 9.2, 0.0, 20.5, 31.0, 2.0]  # ms
    dir_field_offsets = [11.0, 18.8, 37.8, 13.8, 0.8, 19.3, 16.0, 0.0]  # ms

    sac_data = load_sac_rho_data(basest)
    sac_metrics = get_sac_metrics(sac_data)
    tuning = analyze_tree(sac_data, dir_labels, pref=0, thresh=-57)

    sac_thetas = get_sac_thetas(sac_data)
    sac_deltas = get_sac_deltas(sac_thetas)

    ttx = True
    if ttx:
        rec_locs = sac_data["0.00"][0]["dendrites"]["locs"]
        syn_locs = sac_data["0.00"][0]["syn_locs"]
        syn_rec_lookups = get_syn_rec_lookups(rec_locs, syn_locs)
        post_syn_avg_tuning = get_postsyn_avg_tuning(tuning, syn_rec_lookups)
        theta_diffs = plot_theta_diff_tuning_scatters(post_syn_avg_tuning, sac_deltas)
        theta_diff_bins = plot_theta_diff_vs_abs_theta(post_syn_avg_tuning, sac_deltas)
        save_fig(theta_diffs, "theta_diff_tuning_net")
        save_fig(theta_diff_bins, "theta_diff_bins_net")

    polars = sac_rho_polars(
        sac_metrics,
        dir_labels,
        net_shadows=True,
        save=True,
        save_pth=fig_pth,
        save_ext="svg",
    )

    # make_sac_rec_gifs(
    #     fig_pth, sac_data, dir_labels, rhos=[], rec_key="trans_Vm",
    #     sample_inter=10, gif_inter=50,
    # )

    # make_sac_tuning_gifs(
    #     fig_pth, sac_data, dir_labels, rhos=[], rec_key="spk_Vm",
    #     sample_inter=10, gif_inter=50,
    # )
    for net_idx in range(len(sac_data["0.00"])):
        tree = plot_tree_tuning(tuning, net_idx, dsi_mul=500)
        save_fig(tree, "tree_tuning_net%i" % net_idx)

    violins = sac_rho_violins(sac_metrics, dir_labels)
    scatter = ds_scatter(tuning)
    rasters = spike_rasters(
        sac_data,
        dir_labels,
        rho="1.00",
        bin_ms=50,
        offsets=dir_field_offsets,
        colour="black",
        spike_vmax=160,
    )
    evol = time_evolution(sac_data, dir_labels, kernel_var=45)

    if 1:
        save_fig(violins, "selectivity_violins")
        save_fig(tree, "tree_tuning")
        save_fig(scatter, "ds_scatter")
        save_fig(rasters, "spike_rasters")
        save_fig(evol, "spike_evolution")

    plt.show()
