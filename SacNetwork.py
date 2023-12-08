from neuron import h

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import json

from scipy.optimize import curve_fit

from modelUtils import rotate, wrap_180, wrap_360
from general_utils import clean_axes
from scipy.optimize import bisect


class SacNetwork:
    def __init__(
        self,
        syn_locs,
        dir_pr,
        rho,
        uniform_dist,
        shared_var,
        theta_vars,
        gaba_coverage,
        dirs,
        np_rng,
        offset=30,
        theta_mode="PN",
        cell_pref=0,
        n_plexus_ach=0,
        plexus_share=None,
        stacked_plex=False,
        fix_rho_mode=False,
        plexus_syn_mode="all",
    ):
        self.syn_locs = syn_locs  # xy coord ndarray of shape (N, 2)
        self.dir_pr = dir_pr  # {"E": {"null": _, "pref": _} ...}
        self.rho = rho
        self.uniform_dist = uniform_dist  # {0: bool, 1: bool}
        self.shared_var = shared_var
        self.theta_vars = theta_vars  # {"E": _, "I": _}
        self.gaba_coverage = gaba_coverage
        self.dir_labels = dirs
        self.offset = offset
        self.np_rng = np_rng
        self.theta_mode = theta_mode  # TODO: Make a theta param dict...
        self.cell_pref = cell_pref
        self.n_plexus_ach = n_plexus_ach
        self.stacked_plex = stacked_plex
        self.fix_rho_mode = fix_rho_mode
        self.plexus_syn_mode = plexus_syn_mode
        # polyfit params obtained in sacnet_angle_sanity.ipynb
        # describing the relationship between input rho and the resulting
        # circular correlation coefficient (astropy.stats.circstats.circcorrcoef),
        # which is somewhat logarithmic in appearance despite not well fit by same.
        # The corrcoef rises faster than rho initially (inflating actual
        # correlation), and slows as rho approaches 1.
        # These params are used with bisect to find a "fixed rho" that will
        # target the same corrcoef (so rho ~= corr).
        self.rho_corr_poly_params = np.array(
            [-1.20091016, 1.26930728, 0.92224309, 0.0029899]
        )
        rho_corr_poly = np.poly1d(self.rho_corr_poly_params)
        self.fixed_rho = (
            rho
            if rho == 0 or rho == 1
            else bisect(lambda x: rho_corr_poly(x) - rho, 0, 1)
        )

        if n_plexus_ach > 0:
            if plexus_share is None:
                self.plex_prob_mod = 1 / (n_plexus_ach + 1)
                self.syn_prob_mod = self.plex_prob_mod
            else:
                self.plex_prob_mod = plexus_share / n_plexus_ach
                self.syn_prob_mod = 1.0 - plexus_share
        else:
            self.syn_prob_mod = 1.0
            self.plex_prob_mod = 0.0

        if self.theta_mode == "PN":
            self.theta_picker = self.theta_picker_PN
        elif self.theta_mode == "cardinal":
            self.theta_picker = self.theta_picker_cardinal
        elif self.theta_mode == "uniform":
            self.theta_picker = self.theta_picker_uniform
        elif self.theta_mode == "uniform_post":
            self.theta_picker = self.theta_picker_uniform_post
        elif self.theta_mode == "experimental":
            self.theta_picker = self.theta_picker_exp
        else:
            raise Exception("Invalid theta_mode.")

        self.build()

    def build(self):
        self.gaba_here = []  # bool, whether GABA is at a synapse
        self.thetas = {"E": [], "I": []}
        self.bp_locs = {
            "E": np.zeros_like(self.syn_locs),
            "I": np.zeros_like(self.syn_locs),
        }
        self.probs = {"E": [], "I": []}  # shapes: (numSyns, dirs)

        ach_pref, ach_null = self.dir_pr["E"]["pref"], self.dir_pr["E"]["null"]
        if self.n_plexus_ach > 0:
            self.thetas["PLEX"] = []  # shape: (n_syns, n_plex)
            self.bp_locs["PLEX"] = np.zeros((len(self.syn_locs), self.n_plexus_ach, 2))
            self.probs["PLEX"] = []  # shape: (n_syns, n_dirs, n_plex)

        # generator for picking determining GABA presence and dendritic angles

        for i, (syn_x, syn_y) in enumerate(self.syn_locs):
            # determine gaba presence and select thetas (append to lists)
            e_theta, has_gaba, i_theta = self.theta_picker()
            self.thetas["E"].append(e_theta)
            self.thetas["I"].append(i_theta)
            self.gaba_here.append(has_gaba)

            for t in ["E", "I"]:
                # Coordinates of current SAC dendrites INPUT location that
                # governs stimulus offset of the output on to the DSGC.
                self.bp_locs[t][i] = self.locate_bp(syn_x, syn_y, self.thetas[t][-1])

                # Probabilities of release based on similarity of dendrite
                # angle with direction of stimulus.
                pref, null = self.dir_pr[t]["pref"], self.dir_pr[t]["null"]
                self.probs[t].append(self.compute_probs(pref, null, self.thetas[t][-1]))

            if self.n_plexus_ach > 0:
                ach_prob = self.probs["E"][i]
                n = self.n_plexus_ach
                if (self.plexus_syn_mode == "only_pref" and has_gaba) or (
                    self.plexus_syn_mode == "only_null" and not has_gaba
                ):
                    self.probs["PLEX"].append(np.zeros((len(self.dir_labels), n)))
                    self.bp_locs["PLEX"][i] = np.full((n, 2), np.nan)
                    self.thetas["PLEX"].append([np.nan] * n)
                else:
                    self.probs["E"][i] = ach_prob * self.syn_prob_mod
                    if self.stacked_plex:
                        self.thetas["PLEX"].append([e_theta] * n)
                        self.bp_locs["PLEX"][i] = np.repeat(
                            np.expand_dims(self.bp_locs["E"][i], 0), n, axis=0
                        )
                        self.probs["PLEX"].append(
                            np.repeat(
                                np.expand_dims(ach_prob * self.plex_prob_mod, 0),
                                n,
                                axis=0,
                            ).T
                        )
                    else:
                        thetas = [
                            self.np_rng.uniform() * 360.0
                            for _ in range(self.n_plexus_ach)
                        ]
                        self.bp_locs["PLEX"][i] = np.array(
                            [self.locate_bp(syn_x, syn_y, theta) for theta in thetas]
                        )
                        self.probs["PLEX"].append(
                            np.array(
                                [
                                    self.compute_probs(
                                        ach_pref,
                                        ach_null,
                                        theta,
                                    )
                                    * self.plex_prob_mod
                                    for theta in thetas
                                ]
                            ).T
                        )
                        self.thetas["PLEX"].append(thetas)

        self.origin = self.find_origin()
        self.gaba_here = np.array(self.gaba_here)

        for t in self.thetas.keys():
            self.probs[t] = np.nan_to_num(self.probs[t])
            self.thetas[t] = np.array(self.thetas[t])

        # theta difference used by DSGC model to scale down rho
        wrap_180_vec = np.vectorize(wrap_180)
        self.deltas = wrap_180_vec(self.thetas["E"] - self.thetas["I"])
        all_thetas = [self.thetas["E"].reshape(-1, 1)]
        if self.n_plexus_ach > 0:
            all_thetas.append(self.thetas["PLEX"])
        all_thetas.append(self.thetas["I"].reshape(-1, 1))
        theta_matrix = np.concatenate(all_thetas, axis=1)
        self.delta_matrix = wrap_180_vec(
            np.expand_dims(theta_matrix, 1) - np.expand_dims(theta_matrix, 2)
        )

    def theta_picker_PN(self):
        # determine whether there is GABA present at this synapse
        gaba_here = self.np_rng.uniform(0, 1) < self.gaba_coverage

        # inner portion of "two-stage" randomness
        if self.uniform_dist[0]:
            shared = self.np_rng.uniform(-self.shared_var, self.shared_var)
        else:
            shared = self.np_rng.normal(0, 1) * self.shared_var

        # pseudo-random numbers for angle determination (outer portion)
        pick = self.get_outer_picks(gaba_here)
        base = 180 if gaba_here else 0
        theta = lambda t: wrap_360(
            pick[t] * self.theta_vars[t] + shared + base + self.cell_pref
        )

        return theta("E"), gaba_here, (theta("I") if gaba_here else np.NaN)

    def theta_picker_cardinal(self):
        base = (self.np_rng.uniform(0, 1) // 0.25) * 90

        if base == 180:
            gaba_prob = 2 * self.gaba_coverage * 1
        elif base == 90 or base == 270:
            gaba_prob = 2 * self.gaba_coverage * 0.25
        else:
            gaba_prob = 2 * self.gaba_coverage * 0

        # pseudo-random numbers for angle determination (outer portion)
        gaba_here = self.np_rng.uniform(0, 1) < gaba_prob
        pick = self.get_outer_picks(gaba_here)
        theta = lambda t: wrap_360(pick[t] * self.theta_vars[t] + base)

        return theta("E"), gaba_here, (theta("I") if gaba_here else np.NaN)

    def theta_picker_uniform(self):
        p = 0
        n = 1
        base = self.np_rng.uniform(-180, 180)
        gaba_prob = p + (n - p) * (1 - 1 / (1 + np.exp(np.abs(base) - 96) / 25))

        pt = 0.45
        m0 = 2
        m1 = 82
        if self.gaba_coverage > pt:
            g = (m0 * pt) + max(0, self.gaba_coverage - pt) * m1
        else:
            g = self.gaba_coverage * m0

        gaba_prob *= g
        gaba_here = self.np_rng.uniform(0, 1) < gaba_prob
        pick = self.get_outer_picks(gaba_here)
        theta = lambda t: wrap_360(pick[t] * self.theta_vars[t] + base + self.cell_pref)

        return theta("E"), gaba_here, (theta("I") if gaba_here else np.NaN)

    def theta_picker_uniform_post(self):
        p = 0.05
        n = 1
        base = self.np_rng.uniform(-180, 180)

        # pseudo-random numbers for angle determination (outer portion)
        pick = self.get_outer_picks(True)

        e_theta = wrap_360(pick["E"] * self.theta_vars["E"] + base + self.cell_pref)
        i_theta = wrap_360(pick["I"] * self.theta_vars["I"] + base + self.cell_pref)

        d = np.abs(i_theta - (i_theta // 180) * 360)
        gaba_prob = p + (n - p) * (1 - 0.98 / (1 + np.exp(d - 120) / 25))
        gaba_prob *= 2 * self.gaba_coverage  # NOTE: Approximate, fix if used...
        gaba_here = self.np_rng.uniform(0, 1) < gaba_prob

        return e_theta, gaba_here, (i_theta if gaba_here else np.NaN)

    def get_outer_picks(self, correlate):
        pick = {}
        for t in ["E", "I"]:
            if self.uniform_dist[1]:
                pick[t] = self.np_rng.uniform(-1, 1)
            else:
                pick[t] = self.np_rng.normal(0, 1)

        if correlate:
            pick["E"] = pick["I"] * self.rho + pick["E"] * np.sqrt(1 - self.rho**2)
        return pick

    def theta_picker_exp(self):
        s = self.gaba_coverage - 0.5
        p, n = (0, (1.0 + s * 2)) if s < 0 else (s * 2, 1)

        base = self.np_rng.uniform(-180, 180)
        # gaba_prob = p + (n - p) * (1 - 1 / (1 + np.exp((np.abs(base) - 160) * 0.04)))
        # gaba_prob = p + (n - p) * (1 - 1 / (1 + np.exp((np.abs(base) - 90) * 0.04)))
        gaba_prob = p + (n - p) * (1 - 1 / (1 + np.exp((np.abs(base) - 90) * 0.1)))
        # gaba_prob = p + (n - p) * (1 - 1 / (1 + np.exp((np.abs(base) - 90) * 0.12)))
        # var = 180 * np.sqrt(1 - self.rho**2)  # type:ignore
        # var = 180 * np.sqrt(1 - self.rho)  # type:ignore
        rho = self.fixed_rho if self.fix_rho_mode else self.rho
        var = 180 * (1 - rho)  # type:ignore

        gaba_here = self.np_rng.uniform(0, 1) < gaba_prob
        i_theta = base + self.cell_pref
        e_theta = i_theta + self.np_rng.uniform(-var, var)

        return (
            wrap_360(e_theta),
            gaba_here,
            (wrap_360(i_theta) if gaba_here else np.NaN),
        )

    def find_origin(self):
        all_x = np.concatenate(
            [self.bp_locs["E"][:, 0], self.bp_locs["I"][:, 0]], axis=0
        )
        all_x = all_x[~np.isnan(all_x)]
        left_x, right_x = np.min(all_x), np.max(all_x)

        allY = np.concatenate(
            [self.bp_locs["E"][:, 1], self.bp_locs["I"][:, 1]], axis=0
        )
        allY = allY[~np.isnan(allY)]
        bot_y, top_y = np.min(allY), np.max(allY)

        return (left_x + (right_x - left_x) / 2, bot_y + (top_y - bot_y) / 2)

    def locate_bp(self, syn_x, syn_y, theta):
        return np.array(
            [
                syn_x - self.offset * np.cos(np.deg2rad(theta)),
                syn_y - self.offset * np.sin(np.deg2rad(theta)),
            ]
        )

    def compute_probs(self, pref, null, theta):
        probs = []
        for d in self.dir_labels:
            pr = np.abs(theta - d)
            pr = np.abs(pr - 360) if pr > 180 else pr
            # pr = pref + (null - pref) * (1 - 1 / (1 + np.exp((pr - 90) * 0.05)))
            pr = pref + (null - pref) * (1 - 1 / (1 + np.exp((pr - 90) * 0.1)))
            probs.append(pr)
        return np.array(probs)

    def get_syn_loc(self, trans, num, rotation):
        if trans != "PLEX":
            x, y = self.bp_locs[trans][num, 0], self.bp_locs[trans][num, 1]
        else:
            x, y = self.bp_locs[trans][num, :, 0], self.bp_locs[trans][num, :, 1]
        x, y = x, y if rotation == 0 else rotate(self.origin, x, y, rotation)
        return {"x": x, "y": y}

    def plot_dends(self, stim_angle=None, separate=False, cmap="jet"):
        if separate:
            fig, ax = plt.subplots(2, figsize=(8, 12))
        else:
            fig, ax = plt.subplots(1, figsize=(8, 8))

        for (syn_x, syn_y), (ach_x, ach_y), (gaba_x, gaba_y), ePr, iPr in zip(
            self.syn_locs,
            self.bp_locs["E"],
            self.bp_locs["I"],
            self.probs["E"],
            self.probs["I"],
        ):
            if not separate:
                ax.plot([syn_x, ach_x], [syn_y, ach_y], c="g", alpha=0.5)
                ax.plot([syn_x, gaba_x], [syn_y, gaba_y], c="m", alpha=0.5)
            else:
                ax[0].plot([syn_x, ach_x], [syn_y, ach_y], c="g", alpha=0.5)
                ax[1].plot([syn_x, gaba_x], [syn_y, gaba_y], c="m", alpha=0.5)

            if stim_angle is not None:
                angle_idx = np.argwhere(self.dir_labels == np.array(stim_angle))[0][0]
                colors = [plt.get_cmap(cmap)(1.0 * i / 10) for i in range(10)]
                eClr = colors[int(ePr[angle_idx] / 0.1)]
                a = ax[0] if separate else ax
                a.scatter(ach_x, ach_y, c=[eClr], s=120, alpha=0.5)

                if not np.isnan(gaba_x):
                    iClr = colors[int(iPr[angle_idx] / 0.1)]
                    a = ax[1] if separate else ax
                    a.scatter(gaba_x, gaba_y, marker="v", c=[iClr], s=120, alpha=0.5)
            else:
                if not separate:
                    ax.scatter(ach_x, ach_y, c="g", s=120, alpha=0.5)
                    ax.scatter(gaba_x, gaba_y, marker="v", c="m", s=120, alpha=0.5)
                else:
                    ax[0].scatter(ach_x, ach_y, c="g", s=120, alpha=0.5)
                    ax[1].scatter(gaba_x, gaba_y, marker="v", c="m", s=120, alpha=0.5)

        plt.show()

    def plot_dends_overlay(
        self,
        dsgc,
        dsgc_alpha=0.35,
        sac_alpha=0.7,
        x_off=-2,
        y_off=-41,
        px_per_um=2.55,
        stim_angle=None,
        separate=False,
        n_syn=None,
        cmap="plasma",
    ):
        if separate:
            fig, ax = plt.subplots(2, figsize=(8, 12))
        else:
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax = [ax]

        # flip y-axis of image to match it up with coordinate system
        dsgc = np.flip(dsgc, axis=0)

        # offset whitespace and convert um locations to px
        syn_xs = self.syn_locs[:, 0] * px_per_um + x_off
        syn_ys = self.syn_locs[:, 1] * px_per_um + y_off
        ach_xs = np.array(self.bp_locs["E"][:, 0]) * px_per_um + x_off
        ach_ys = np.array(self.bp_locs["E"][:, 1]) * px_per_um + y_off
        gaba_xs = np.array(self.bp_locs["I"][:, 0]) * px_per_um + x_off
        gaba_ys = np.array(self.bp_locs["I"][:, 1]) * px_per_um + y_off

        for a in ax:
            a.imshow(dsgc, alpha=dsgc_alpha, cmap="gray")
            a.set_xlim(-70, 600)
            a.set_ylim(-70, 600)

        idxs = np.random.choice(
            len(syn_xs), size=len(syn_xs) if n_syn is None else n_syn, replace=False
        )
        for i in idxs:
            if not separate:
                ax[0].plot(
                    [syn_xs[i], ach_xs[i]],
                    [syn_ys[i], ach_ys[i]],
                    c="g",
                    alpha=sac_alpha,
                )
                ax[0].plot(
                    [syn_xs[i], gaba_xs[i]],
                    [syn_ys[i], gaba_ys[i]],
                    c="m",
                    alpha=sac_alpha,
                )
            else:
                ax[0].plot(
                    [syn_xs[i], ach_xs[i]],
                    [syn_ys[i], ach_ys[i]],
                    c="g",
                    alpha=sac_alpha,
                )
                ax[1].plot(
                    [syn_xs[i], gaba_xs[i]],
                    [syn_ys[i], gaba_ys[i]],
                    c="m",
                    alpha=sac_alpha,
                )

            if stim_angle is not None:
                angle_idx = np.argwhere(self.dir_labels == np.array(stim_angle))[0][0]
                colors = [plt.get_cmap(cmap)(1.0 * i / 100) for i in range(100)]
                eClr = colors[int(self.probs["E"][i][angle_idx] / 0.01)]
                scat = ax[0].scatter(
                    ach_xs[i], ach_ys[i], c=[eClr], s=120, alpha=sac_alpha
                )

                if not np.isnan(gaba_xs[i]):
                    iClr = colors[int(self.probs["I"][i][angle_idx] / 0.01)]
                    ax[-1].scatter(
                        gaba_xs[i],
                        gaba_ys[i],
                        marker="v",
                        c=[iClr],
                        s=120,
                        alpha=sac_alpha,
                    )

            else:
                if not separate:
                    ax[0].scatter(ach_xs[i], ach_ys[i], c="g", s=120, alpha=sac_alpha)
                    ax[0].scatter(
                        gaba_xs[i],
                        gaba_ys[i],
                        marker="v",
                        c="m",
                        s=120,
                        alpha=sac_alpha,
                    )
                else:
                    ax[0].scatter(ach_xs[i], ach_ys[i], c="g", s=120, alpha=sac_alpha)
                    ax[1].scatter(
                        gaba_xs[i],
                        gaba_ys[i],
                        marker="v",
                        c="m",
                        s=120,
                        alpha=sac_alpha,
                    )

        if stim_angle is not None:

            cbar = fig.colorbar(
                ScalarMappable(Normalize(vmin=0.0, vmax=1.0), cmap=cmap),
                ax=ax[0],
                orientation="horizontal",
                pad=0.0,
                shrink=0.75,
            )
            cbar.ax.tick_params(labelsize=12.0)
            cbar.set_label("Release Probability", fontsize=12)

        ach_circ = ax[0].scatter([], [], c="g", s=120)
        gaba_tri = ax[0].scatter([], [], c="m", marker="v", s=120)
        fig.legend(
            [ach_circ, gaba_tri],
            ["ACh", "GABA"],
            ncol=1,
            frameon=False,
            fontsize=12,
            handlelength=2,
            loc="lower left",
            bbox_to_anchor=(0.15, 0.25),
            borderpad=1,
            handletextpad=1,
            title_fontsize=12,
        )

        clean_axes(ax, remove_spines=["left", "right", "top", "bottom"])
        fig.tight_layout()
        plt.show()

    def hist_angle(self, bins=[8, 12, 16]):
        # drop NaNs from GABA thetas first
        iThetas = np.array(self.thetas["I"])[np.array(self.gaba_here)]
        fig, axes = plt.subplots(1, len(bins))

        for j, numBins in enumerate(bins):
            binAx = [i * 360 / numBins for i in range(numBins)]
            axes[j].hist(self.thetas["E"], bins=binAx, color="r", alpha=0.5)
            axes[j].hist(iThetas, bins=binAx, color="c", alpha=0.5)
            axes[j].set_title("Bin Size: %.1f" % (360 / numBins))
            axes[j].set_xlabel("SAC Dendrite Angle")

        plt.show()

    def get_wiring_dict(self):
        """Return a dict ready to be packed in to an HDF5 archive. Parameters
        are formatted as a json, generated values are store separately."""
        param_keys = {"dir_pr", "rho", "theta_vars", "gaba_coverage", "offset", "seed"}

        net = {
            "params": json.dumps(
                {k: v for k, v in self.__dict__.items() if k in param_keys}
            ),
            "wiring": {
                k: v
                for k, v in self.__dict__.items()
                if k in {"gaba_here", "bp_locs", "thetas", "probs"}
            },
        }

        return net
