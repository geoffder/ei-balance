# science/math libraries
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# general libraries
import os  # make folders for file output
import h5py as h5
import json

# local libraries
from modelUtils import rotate, scale_180_from_360


def unpack_hdf(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
    return {
        k: v[()] if type(v) is h5._hl.dataset.Dataset else unpack_hdf(v)
        for k, v in group.items()
    }


def spike_transform(vm, kernel_sz, kernel_var, thresh=0):
    """Trying out a dead simple gaussian kernel transform to apply to Vm.
    Applies convolution to last dimension, all other dimensions should just
    be trials/locations/directions etc (e.g. data is just 1D time-series).
    Soma Vm is stored in shape (Trials, Directions, Time).
    """
    kernel = signal.gaussian(kernel_sz, kernel_var)
    og_shape = vm.shape
    vm = vm.reshape(-1, vm.shape[-1])

    spks, conv = np.zeros_like(vm), np.zeros_like(vm)
    
    for i, rec in enumerate(vm):
        idxs, _ = signal.find_peaks(rec, height=thresh)
        spks[i][idxs] = 1
        conv[i] = np.convolve(spks[i], kernel, mode="same")

    return conv.reshape(og_shape)


def gauss_conv(vm, kernel_sz, kernel_var):
    """Gaussian convolution of input recordings (without spike detection step).
    """
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
        
    xsums = np.multiply(data, np.cos(rads)).sum(axis=0)
    ysums = np.multiply(data, np.sin(rads)).sum(axis=0)
    DSi = np.squeeze(
        np.sqrt(xsums ** 2 + ysums ** 2)
        / (np.sum(data, axis=0) + .000001)
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
        [ox + np.cos(theta - np.pi/2) * 200,
         ox + np.cos(theta + np.pi/2) * 200]
    )
    y_ends = np.array(
        [oy + np.sin(theta - np.pi/2) * 200,
         oy + np.sin(theta + np.pi/2) * 200]
    )

    x_step = stim["dt"] * stim["speed"] * np.cos(theta)
    y_step = stim["dt"] * stim["speed"] * np.sin(theta)
    bar_x = np.array([x_ends + x_step * t for t in range(0, stim["n_pts"])])
    bar_y = np.array([y_ends + y_step * t for t in range(0, stim["n_pts"])])

    return bar_x, bar_y


def activity_gif(
        data, locs, fname, vmin=-63, vmax=10, sample_inter=25, rec_dt=.1,
        gif_inter=150, dpi=100,
):
    """Create animated gif depicting dendritic activity in response to a
    simulated visual stimulus. A bar depicting the edge of the stimulus is
    overlaid, and the somatic Vm recording is included underneath."""
    fig, ax = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [.8, .2]}, figsize=(5, 6)
    )

    scat = ax[0].scatter(
        locs[0], locs[1], c=data["tree"][:, 0], vmin=vmin, vmax=vmax
    )

    bar_x, bar_y = generate_bar_sweep(data["stim"])
    stim = ax[0].plot(bar_x[0], bar_y[0], c='black', linewidth=3)
    ax[0].set_xlim(-30, 230)
    ax[0].set_ylim(-30, 260)

    time = np.linspace(0, len(data["soma"]) * rec_dt, len(data["soma"]))
    line = ax[1].plot(time[0], data["soma"][0])
    ax[1].set_xlim(time.min(), time.max())
    ax[1].set_ylim(-70, 30)

    rate_diff = data["soma"].shape[0] // data["tree"].shape[1]

    def update(t):
        scat.set_array(data["tree"][:, t])
        line[0].set_xdata(time[:t * rate_diff])
        line[0].set_ydata(data["soma"][:t * rate_diff])
        stim[0].set_xdata(bar_x[t * rate_diff])
        stim[0].set_ydata(bar_y[t * rate_diff])
        return scat, line, stim

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, data["tree"].shape[1], sample_inter),
        interval=gif_inter
    )
    anim.save(fname, dpi=dpi, writer='imagemagick')
    plt.close()


def make_sac_rec_gifs(
        pth, data, dirs, rec_key="Vm", rhos=[], net=0, trial=0, sample_inter=25,
        gif_inter=150, dpi=100
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
                }
            }
            activity_gif(
                gif_data,
                rec_locs,
                gif_pth + "rho%s_tr%i_dir%i_%s.gif" % (rho, trial, d, rec_key),
                vmin=vmin,
                vmax=vmax,
                sample_inter=sample_inter,
                gif_inter=gif_inter,
                dpi=dpi
            )

            
def tuning_gif(data, locs, fname, dsi_mul=250, sample_inter=25, rec_dt=.1,
               gif_inter=150, dpi=100):
    """Create animated gif depicting directional preference in response to a
    simulated visual stimulus."""
    data["tree"]["DSi"] *= dsi_mul
    
    fig, ax = plt.subplots(
        3, 1, gridspec_kw={"height_ratios": [.7, .15, .15]}, figsize=(6, 7)
    )
    ax[1].set_ylabel("theta (°)", size=14)
    ax[2].set_ylabel("DSi", size=14)
    ax[1].set_xticklabels([])
    ax[2].set_xlabel("Time", size=14)

    scat = ax[0].scatter(
        locs[0], locs[1],
        c=data["tree"]["theta"][:, 0],
        s=data["tree"]["DSi"][:, 0],
        vmin=data["tree"]["min_theta"],
        vmax=data["tree"]["max_theta"]
    )
    time = np.linspace(
        0, len(data["soma"]["DSi"]) * rec_dt, len(data["soma"]["DSi"]))
    theta_line = ax[1].plot(time[0], data["soma"]["theta"][0])
    DSi_line = ax[2].plot(time[0], data["soma"]["DSi"][0])

    ax[1].set_xlim(time.min(), time.max())
    ax[2].set_xlim(time.min(), time.max())
    ax[1].set_ylim(data["soma"]["theta"].min(), data["soma"]["theta"].max(),)
    ax[2].set_ylim(data["soma"]["DSi"].min(), data["soma"]["DSi"].max(),)
    
    rate_diff = data["soma"]["DSi"].shape[0] // data["tree"]["DSi"].shape[1]

    # draw legend mapping DSi -> size with min/max examples.
    mn = ax[0].scatter(
        [], [], c="black", s=data["tree"]["min_DSi"]*dsi_mul, edgecolors="none")
    mx = ax[0].scatter(
        [], [], c="black", s=data["tree"]["max_DSi"]*dsi_mul, edgecolors="none")
    labels = ["%.2f" % data["tree"]["%s_DSi" % m] for m in ["min", "max"]]
    legend = fig.legend(
        [mn, mx], labels, ncol=2, frameon=True, fontsize=12,
        handlelength=2, loc="upper right", borderpad=1, handletextpad=1,
        title="DSi Bounds", title_fontsize=18,
    )
    fig.colorbar(scat, ax=ax[0], orientation="vertical", pad=.1)

    def update(t):
        scat.set_array(data["tree"]["theta"][:, t])
        scat.set_sizes(data["tree"]["DSi"][:, t])
        theta_line[0].set_xdata(time[:t * rate_diff])
        theta_line[0].set_ydata(data["soma"]["theta"][:t * rate_diff])
        DSi_line[0].set_xdata(time[:t * rate_diff])
        DSi_line[0].set_ydata(data["soma"]["DSi"][:t * rate_diff])
        return scat, theta_line, DSi_line

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, data["tree"]["DSi"].shape[1], sample_inter),
        interval=gif_inter
    )
    anim.save(fname, dpi=dpi, writer='imagemagick')
    plt.close()

            
def make_sac_tuning_gifs(
        pth, data, dirs, rec_key="Vm", rhos=[], net=0, thresh=None,
        sample_inter=25, gif_inter=150, dpi=100
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
            recs = gauss_conv(
                data[rho][net]["dendrites"]["Vm"], 500, 15
            )
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
                dpi=dpi
            )


def polar_plot(metrics, dirs, radius=None, net_shadows=False, title="",
               save=False, save_pth=""):
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
    
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="polar")

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
    ax.plot(
        [avg_theta, avg_theta], [0.0, avg_DSi * peak],
        color=".0", linewidth=2
    )

    # misc settings
    if radius is None:
        ax.set_rmax(peak)
    else:
        ax.set_rmax(radius)
    ax.set_rticks([peak])
    ax.set_rlabel_position(45)
    ax.set_thetagrids([0, 90, 180, 270])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=15)
    sub = "DSi = %.2f :: θ = %.2f\n  σ = %.2f :: σ = %.2f" % (
        avg_DSi,
        np.degrees(avg_theta),
        np.std(DSis),
        np.std(np.vectorize(scale_180_from_360)(np.degrees(thetas - avg_theta)))
    )
    ax.set_title("%s\n%s" % (title, sub), size=15)

    if save:
        fig.savefig(
            "%spolar_%s.png" % (save_pth, title.replace(" ", "_")),
            bbox_inches="tight"
        )

    return fig


def load_sac_rho_data(pth, prefix="sac_rho"):
    """Collect data from each of the hdf5 archives located in the target folder
    representing rho conditions and their trials and store them in a dict."""
    all_exps = [
        f for f in os.listdir(pth)
        if prefix in f and os.path.isfile(os.path.join(pth, f))
    ]

    # assumes rho is .2f precision following prefix. Should make less
    # hacky at some point.
    fnames = {
        r: [f for f in all_exps if r in f]
        for r in set([n[len(prefix):len(prefix) + 4] for n in all_exps])
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

    
def sac_rho_polars(metrics, dirs, save=False, net_shadows=False, save_pth=""):
    """Collect pre-calculated metrics from given data dict and feed them in to
    polar_plot function for each sac angle rho level included in the data."""
    max_spikes = np.max([r["spikes"] for r in metrics.values()])

    polar_figs = [
        polar_plot(
            m, dirs, title="rho " + r, radius=max_spikes,
            net_shadows=net_shadows, save=save, save_pth=save_pth
        )
        for r, m in metrics.items()
    ]

    return polar_figs


def sac_rho_violins(metrics, dir_labels, viol_inner="box"):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(6, 10))

    labels = [k for k, v in metrics.items() for _ in range(v["DSis"].size)]
    dsis = np.concatenate([r["DSis"].flatten() for r in metrics.values()])
    sns.violinplot(labels, dsis, inner=viol_inner, ax=ax[0])
    ax[0].set_ylabel("DSi", size=14)
    ax[0].set_ylim(0)

    # for each rho level, reshape to (Directions, Trials * Nets)
    dir_spks = {
        k: v["spikes"].transpose().reshape(v["spikes"].shape[-1], -1)
        for k, v in metrics.items()
    }
    idxs = [dir_labels.index(d) for d in [135, 180, 225]]

    for a, idx in zip(ax[1:], idxs):
        spks = np.concatenate([r[idx] for r in dir_spks.values()])
        sns.violinplot(labels, spks, inner=viol_inner, ax=a)
        a.set_ylabel("Spikes in %d" % dir_labels[idx], size=14)
        a.set_ylim(-1)
        a.set_yticks(np.arange(0, spks.max() + 2, 2))
        
    ax[-1].set_xlabel("Correlation (rho)", size=14)
    fig.tight_layout()
    
    return fig


def dendritic_ds(tree_recs, dirs, pref=0, thresh=None, bin_pts=None):
    """Expected dendrite recordings are in an array of shape (N, Dirs, Locs, T).
    If passing in an average, include a singleton 0th dimension.
    """
    if bin_pts is not None:
        tree_recs = tree_recs[:, :, :, bin_pts[0]:bin_pts[1]]

    if thresh is not None:
        tree_recs = np.clip(tree_recs, thresh, None) - thresh
    else:
        tree_recs = tree_recs - tree_recs.min()
        
    areas = np.sum(tree_recs, axis=3)    

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
    for rho, nets in data.items():
        tuning[rho] = {}
        for i in nets.keys():
            tree_recs = data[rho][i]["dendrites"][rec_key]
            avg_recs = np.mean(tree_recs, axis=0, keepdims=True)
            tuning[rho][i] = {
                "trials": dendritic_ds(
                    tree_recs, dirs, pref=pref, thresh=thresh, bin_pts=bin_pts),
                "avg": dendritic_ds(
                    avg_recs, dirs, pref=pref, thresh=thresh, bin_pts=bin_pts),
                "locs": data[rho][i]["dendrites"]["locs"],
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
    min_dsi, max_dsi = 1., 0.
    min_theta, max_theta = 180., 0.
    
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
            net["locs"][0], net["locs"][1], s=sz, alpha=.5,
            c=np.abs(net[level]["theta"][trial]), cmap="jet",
            vmin=min_theta, vmax=max_theta,
        )
        ax.set_title("rho = %s" % rho, fontsize=18)

    # draw legend mapping DSi -> size with min/max examples.
    if dsi_size:
        mn = axes[0].scatter(
            [], [], c="black", s=min_dsi*dsi_mul, edgecolors="none")
        mx = axes[0].scatter(
            [], [], c="black", s=max_dsi*dsi_mul, edgecolors="none")
        labels = ["%.2f" % m for m in [min_dsi, max_dsi]]
        legend = fig.legend(
            [mn, mx], labels, ncol=2, frameon=True, fontsize=12,
            handlelength=2, loc="upper right", borderpad=1, handletextpad=1,
            title="DSi Bounds", title_fontsize=18,
        )
    
    fig.colorbar(scatter, ax=axes, orientation="horizontal", pad=.1)
    
    return fig


def ds_scatter(tuning_dict):
    fig, axes = plt.subplots(1, len(tuning_dict), figsize=(12, 6))
    
    # sort dict so rho increases left to right
    for (rho, nets), ax in zip(sorted(tuning_dict.items()), axes):
        pts = {
            m: np.concatenate(
                [n["trials"][m].flatten() for n in nets.values()]
            ) for m in ["DSi", "theta"]
        }
        ax.scatter(pts["DSi"], pts["theta"], c="black", alpha=.1)
        ax.set_xlim(0, .5)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("DSi", size=14)
        ax.set_ylabel("theta (°)", size=14)
        ax.set_title("rho = %s" % rho, fontsize=18)

    return fig


def spike_rasters(data, dirs, net_idx=None, thresh=0, bin_ms=0):
    """Plot spike-time raster for each direction, at each rho level. Spikes are
    pooled across trials, and also networks unless a net_idx is specified.
    Optionally, display binned tuning calculations beneath raster."""
    dt = data["0.00"][0]["params"]["dt"]
    rec_sz = data["0.00"][0]["soma"]["Vm"].shape[-1]
    rel_dirs = [d if d <= 180 else d - 360 for d in dir_labels]

    if bin_ms < 1:
        fig, axes = plt.subplots(
            1, len(data), figsize=(12, 6), sharey="row", squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            3, len(data), figsize=(12, 8), sharex="col", sharey="row",
            gridspec_kw={"height_ratios": [.6, .2, .2]}
        )
        pts_per_bin = int(bin_ms / dt)
        bin_pts = np.array(
            [pts_per_bin * i for i in range(1, rec_sz // pts_per_bin + 1)])
        bin_centres = dt * (bin_pts - pts_per_bin / 2)  # for plotting

    # unzip axes from rows in to column-major organization
    axes = [[a[i] for a in axes] for i in range(len(data))]
    
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
        col[0].eventplot(
            dir_ts.values(), lineoffsets=rel_dirs,
            linelengths=30, alpha=.5,
        )
        
        if bin_ms > 0:
            cum_spks = [np.zeros(len(dir_ts))]
            for p in bin_pts:
                # mark bin margins on raster plot
                col[0].axvline(
                    p * net["params"]["dt"], c="black", linestyle="--", alpha=.5
                )
                # number of spikes before each bin margin (for each direction)
                cum_spks.append(
                    [np.sum(ts < (p * dt)) for ts in dir_ts.values()]
                )
            cum_spks = np.array(cum_spks)
            bin_spks = (cum_spks[1:] - cum_spks[0:-1]).T

            # calculate and plot theta/DSi for each bin
            mean_DSi, mean_theta = calc_tuning(bin_spks, dirs, dir_ax=0)

            style = {"linestyle": "--", "marker": "D"}
            col[1].plot(bin_centres, mean_theta, **style)
            col[2].plot(bin_centres, mean_DSi, **style)

        # shared X settings
        col[0].set_xlim(0, rec_sz * dt)
        col[0].set_title("rho = %s" % rho, fontsize=18)
        col[-1].set_xlabel("Time (ms)", size=14)

    # shared Y settings
    axes[0][0].set_ylabel("Direction (°)", size=14)
    axes[0][0].set_yticks(rel_dirs)
    axes[0][1].set_ylabel("theta (°)", size=14)
    axes[0][1].set_ylim(-180, 180)
    axes[0][2].set_ylabel("DSi", size=14)
    axes[0][2].set_ylim(0, 1)

    fig.tight_layout()
    
    return fig


def time_evolution(data, dirs, kernel_var=30, net_idx=None):
    """Apply a continuous transform to spike somatic spike train and use to
    calculate theta and DSi for each moment in time. Display spike waveforms
    (aggregate mean with per network means) for each direction, with theta and
    DSi below."""
    dt = data["0.00"][0]["params"]["dt"]
    rec_sz = data["0.00"][0]["soma"]["Vm"].shape[-1]
    time = np.linspace(0, rec_sz * dt, rec_sz)

    fig, axes = plt.subplots(
        10, len(data), sharex="col", sharey="row", figsize=(10, 8)
    )
    # unzip axes from rows in to column-major organization
    axes = [[a[i] for a in axes] for i in range(len(data))]

    # sort dict so rho increases left to right
    for (rho, nets), col in zip(sorted(data.items()), axes):
        spk_waves = [
            spike_transform(nets[n]["soma"]["Vm"], 500, kernel_var) 
            for n in nets.keys() if net_idx is None or net_idx == n
        ]
        net_means = np.stack([np.mean(w, axis=0) for w in spk_waves], axis=1)
        mean_waves = np.mean(np.concatenate(spk_waves, axis=0), axis=0)

        # plot average spike waves for each direction
        for ax, ns, wave in zip(col, net_means, mean_waves):
            for n in ns:
                ax.plot(time, n, c="tab:blue", linewidth=.75, alpha=.5)
            ax.plot(time, wave, c="tab:blue", linewidth=2)

        # direction selective tuning for each point in time
        mean_DSi, mean_theta = calc_tuning(mean_waves, dirs, dir_ax=0)
        net_DSis, net_thetas = calc_tuning(net_means, dirs, dir_ax=0)

        for ts in net_thetas:
            col[-2].plot(time, ts, c="tab:blue", linewidth=.75, alpha=.5)
        col[-2].plot(time, mean_theta, c="tab:blue", linewidth=2)
        col[-2].set_ylim(-180, 180)

        for ds in net_DSis:
            col[-1].plot(time, ds, c="tab:blue", linewidth=.75, alpha=.5)
        col[-1].plot(time, mean_DSi, c="tab:blue", linewidth=2)
        col[-1].set_ylim(0, 1)

        col[0].set_title("rho = %s" % rho, fontsize=18)

    ymax = net_means.max()
    for col in axes:
        for ax in col[:-2]:
            ax.set_ylim(0, ymax)
        col[-1].set_xlim(time.min(), time.max())
        col[-1].set_xlabel("Time (ms)", size=14)

    for d, ax in zip(dirs, axes[0]):
        ax.set_ylabel("%i°" % d, rotation=0, labelpad=20, size=14,
                      verticalalignment="center")

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


# TODO: Enable rasters, tuning evolution, and violins to adjust to different
# preferred directions as well.
if __name__ == "__main__":
    basest = "/mnt/Data/NEURONoutput/"
    # basest += "uni_var60_E90_I90_ARM_nonDirGABA/"
    fig_pth = basest + "py_figs/"
    if not os.path.isdir(fig_pth):
        os.mkdir(fig_pth)

    dir_labels = [225, 270, 315, 0, 45, 90, 135, 180]

    sac_data = load_sac_rho_data(basest)
    sac_metrics = get_sac_metrics(sac_data)
    tuning = analyze_tree(sac_data, dir_labels, pref=0, thresh=-58)
    
    polars = sac_rho_polars(sac_metrics, dir_labels, net_shadows=True,
                            save=True, save_pth=fig_pth)

    # make_sac_rec_gifs(
    #     fig_pth, sac_data, dir_labels, rhos=[], rec_key="trans_Vm",
    #     sample_inter=10, gif_inter=50,
    # )

    # make_sac_tuning_gifs(
    #     fig_pth, sac_data, dir_labels, rhos=[], rec_key="spk_Vm",
    #     sample_inter=10, gif_inter=50,
    # )

    violins = sac_rho_violins(sac_metrics, dir_labels)
    tree = plot_tree_tuning(tuning, 4, dsi_mul=500)
    scatter = ds_scatter(tuning)
    rasters = spike_rasters(sac_data, dir_labels, bin_ms=50)
    evol = time_evolution(sac_data, dir_labels, kernel_var=45)
    
    if 1:
        violins.savefig(fig_pth + "selectivity_violins.png", bbox_inches="tight")
        tree.savefig(fig_pth + "tree_tuning.png", bbox_inches="tight")
        scatter.savefig(fig_pth + "ds_scatter.png", bbox_inches="tight")
        rasters.savefig(fig_pth + "spike_rasters.png", bbox_inches="tight")
        evol.savefig(fig_pth + "spike_evolution.png", bbox_inches="tight")

    plt.show()

