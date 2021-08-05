from neuron import h, gui

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import json

from scipy.optimize import curve_fit

from modelUtils import rotate, wrap_180, wrap_360
from general_utils import clean_axes


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
        offset=30,
        initial_seed=0,
        theta_mode="PN",
        cell_pref=0,
    ):
        self.syn_locs = syn_locs  # {"X": [], "Y": []}
        self.dir_pr = dir_pr  # {"E": {"null": _, "pref": _} ...}
        self.rho = rho
        self.uniform_dist = uniform_dist  # {0: bool, 1: bool}
        self.shared_var = shared_var
        self.theta_vars = theta_vars  # {"E": _, "I": _}
        self.gaba_coverage = gaba_coverage
        self.dir_labels = dirs
        self.offset = offset
        self.seed = initial_seed
        self.theta_mode = theta_mode  # TODO: Make a theta param dict...
        self.cell_pref = cell_pref

        self.build()

    def build(self):
        self.gaba_here = []  # bool, whether GABA is at a synapse
        self.thetas = {"E": [], "I": []}
        self.bp_locs = {"E": {"X": [], "Y": []}, "I": {"X": [], "Y": []}}
        self.probs = {"E": [], "I": []}  # shapes: (numSyns, dirs)

        # generator for picking determining GABA presence and dendritic angles
        self.rand = h.Random(self.seed)
        self.seed += 1

        for synX, synY in zip(self.syn_locs["X"], self.syn_locs["Y"]):
            # determine gaba presence and select thetas (append to lists)
            if self.theta_mode == "PN":
                self.theta_picker_PN()
            elif self.theta_mode == "cardinal":
                self.theta_picker_cardinal()
            elif self.theta_mode == "uniform":
                self.theta_picker_uniform()
            elif self.theta_mode == "uniform_post":
                self.theta_picker_uniform_post()
            else:
                raise Exception("Invalid theta_mode.")

            for t in ["E", "I"]:
                # Coordinates of current SAC dendrites INPUT location that
                # governs stimulus offset of the output on to the DSGC.
                self.bp_locs[t]["X"].append(
                    synX - self.offset * np.cos(np.deg2rad(self.thetas[t][-1]))
                )
                self.bp_locs[t]["Y"].append(
                    synY - self.offset * np.sin(np.deg2rad(self.thetas[t][-1]))
                )

                # Probabilities of release based on similarity of dendrite
                # angle with direction of stimulus.
                prs = []
                pref, null = self.dir_pr[t]["pref"], self.dir_pr[t]["null"]
                for d in self.dir_labels:
                    prs.append(np.abs(self.thetas[t][-1] - d))
                    prs[-1] = np.abs(prs[-1] - 360) if prs[-1] > 180 else prs[-1]
                    # prs[-1] = pref + (null - pref) * (
                    #     1 - 0.98 / (1 + np.exp((prs[-1] - 91) / 25))
                    # )
                    # NOTE: smoother fall. TESTING
                    # prs[-1] = pref + (null - pref) * (
                    #     1 - 1 / (1 + np.exp((prs[-1] - 90) * 0.75))
                    # )
                    prs[-1] = pref + (null - pref) * (
                        1 - 1 / (1 + np.exp((prs[-1] - 90) * 0.05))
                    )

                self.probs[t].append(prs)

        self.origin = self.find_origin()
        self.gaba_here = np.array(self.gaba_here)

        for t in ["E", "I"]:
            self.probs[t] = np.nan_to_num(self.probs[t])
            self.thetas[t] = np.array(self.thetas[t])
            self.bp_locs[t]["X"] = np.array(self.bp_locs[t]["X"])
            self.bp_locs[t]["Y"] = np.array(self.bp_locs[t]["Y"])

        # theta difference used by DSGC model to scale down rho
        self.deltas = np.vectorize(wrap_180)(self.thetas["E"] - self.thetas["I"])

    def theta_picker_PN(self):
        # determine whether there is GABA present at this synapse
        self.gaba_here.append(self.rand.uniform(0, 1) < self.gaba_coverage)

        # inner portion of "two-stage" randomness
        if self.uniform_dist[0]:
            shared = self.rand.uniform(-self.shared_var, self.shared_var)
        else:
            shared = self.rand.normal(0, 1) * self.shared_var

        # pseudo-random numbers for angle determination (outer portion)
        pick = self.get_outer_picks(self.gaba_here[-1])

        for t in ["E", "I"]:
            if t == "I" and not self.gaba_here[-1]:
                self.thetas["I"].append(np.NaN)
                continue

            base = 180 if self.gaba_here[-1] else 0
            theta = wrap_360(
                pick[t] * self.theta_vars[t] + shared + base + self.cell_pref
            )

            self.thetas[t].append(theta)

    def theta_picker_cardinal(self):
        base = (self.rand.uniform(0, 1) // 0.25) * 90

        if base == 180:
            gaba_prob = 2 * self.gaba_coverage * 1
        elif base == 90 or base == 270:
            gaba_prob = 2 * self.gaba_coverage * 0.25
        else:
            gaba_prob = 2 * self.gaba_coverage * 0

        self.gaba_here.append(self.rand.uniform(0, 1) < gaba_prob)

        # pseudo-random numbers for angle determination (outer portion)
        pick = self.get_outer_picks(self.gaba_here[-1])

        for t in ["E", "I"]:
            if t == "I" and not self.gaba_here[-1]:
                self.thetas["I"].append(np.NaN)
                continue

            theta = wrap_360(pick[t] * self.theta_vars[t] + base)

            self.thetas[t].append(theta)

    def theta_picker_uniform(self):
        p = 0
        n = 1
        base = self.rand.uniform(-180, 180)
        gaba_prob = p + (n - p) * (
            # 1 - 0.98 / (1 + np.exp(np.abs(base) - 91) / 25))
            # 1
            # - 0.98 / (1 + np.exp(np.abs(base) - 96) / 25)
            1
            - 1 / (1 + np.exp(np.abs(base) - 96) / 25)
            # NOTE: see how big a diff this makes. (0.98 vs 1)
            # It does make the histogram sharper due to fewer dends in the 0 bin
        )
        # NOTE: smoother fall. TESTING
        # gaba_prob = p + (n - p) * (1 - 1 / (1 + np.exp((np.abs(base) - 90) * 0.05)))

        pt = 0.45
        m0 = 2
        m1 = 82
        if self.gaba_coverage > pt:
            g = (m0 * pt) + max(0, self.gaba_coverage - pt) * m1
        else:
            g = self.gaba_coverage * m0

        gaba_prob *= g
        self.gaba_here.append(self.rand.uniform(0, 1) < gaba_prob)

        pick = self.get_outer_picks(self.gaba_here[-1])

        for t in ["E", "I"]:
            if t == "I" and not self.gaba_here[-1]:
                self.thetas["I"].append(np.NaN)
                continue

            theta = wrap_360(pick[t] * self.theta_vars[t] + base + self.cell_pref)

            self.thetas[t].append(theta)

    def theta_picker_uniform_post(self):
        p = 0.05
        n = 1
        base = self.rand.uniform(-180, 180)

        # pseudo-random numbers for angle determination (outer portion)
        pick = self.get_outer_picks(True)

        for t in ["E", "I"]:
            theta = wrap_360(pick[t] * self.theta_vars[t] + base + self.cell_pref)

            if t == "I":
                d = np.abs(theta - (theta // 180) * 360)
                gaba_prob = p + (n - p) * (1 - 0.98 / (1 + np.exp(d - 120) / 25))
                gaba_prob *= 2 * self.gaba_coverage  # NOTE: Approximate, fix if used...
                self.gaba_here.append(self.rand.uniform(0, 1) < gaba_prob)
                if not self.gaba_here[-1]:
                    self.thetas["I"].append(np.NaN)
                    continue

            self.thetas[t].append(theta)

    def get_outer_picks(self, correlate):
        pick = {}
        for t in ["E", "I"]:
            if self.uniform_dist[1]:
                pick[t] = self.rand.uniform(-1, 1)
            else:
                pick[t] = self.rand.normal(0, 1)

        if correlate:
            pick["E"] = pick["I"] * self.rho + pick["E"] * np.sqrt(1 - self.rho ** 2)
        return pick

    def find_origin(self):
        allX = np.concatenate([self.bp_locs["E"]["X"], self.bp_locs["I"]["X"]], axis=0)
        allX = allX[~np.isnan(allX)]
        leftX, rightX = np.min(allX), np.max(allX)

        allY = np.concatenate([self.bp_locs["E"]["Y"], self.bp_locs["I"]["Y"]], axis=0)
        allY = allY[~np.isnan(allY)]
        botY, topY = np.min(allY), np.max(allY)

        return (leftX + (rightX - leftX) / 2, botY + (topY - botY) / 2)

    def get_syn_loc(self, trans, num, rotation):
        x, y = rotate(
            self.origin,
            self.bp_locs[trans]["X"][num],
            self.bp_locs[trans]["Y"][num],
            rotation,
        )
        return {"x": x, "y": y}

    def plot_dends(self, stim_angle=None, separate=False, cmap="jet"):
        if separate:
            fig, ax = plt.subplots(2, figsize=(8, 12))
        else:
            fig, ax = plt.subplots(1, figsize=(8, 8))

        for synX, synY, achX, achY, gabaX, gabaY, ePr, iPr in zip(
            self.syn_locs["X"],
            self.syn_locs["Y"],
            self.bp_locs["E"]["X"],
            self.bp_locs["E"]["Y"],
            self.bp_locs["I"]["X"],
            self.bp_locs["I"]["Y"],
            self.probs["E"],
            self.probs["I"],
        ):
            if not separate:
                ax.plot([synX, achX], [synY, achY], c="g", alpha=0.5)
                ax.plot([synX, gabaX], [synY, gabaY], c="m", alpha=0.5)
            else:
                ax[0].plot([synX, achX], [synY, achY], c="g", alpha=0.5)
                ax[1].plot([synX, gabaX], [synY, gabaY], c="m", alpha=0.5)

            if stim_angle is not None:
                angle_idx = np.argwhere(self.dir_labels == np.array(stim_angle))[0][0]
                colors = [plt.get_cmap(cmap)(1.0 * i / 10) for i in range(10)]
                eClr = colors[int(ePr[angle_idx] / 0.1)]
                a = ax[0] if separate else ax
                a.scatter(achX, achY, c=[eClr], s=120, alpha=0.5)

                if not np.isnan(gabaX):
                    iClr = colors[int(iPr[angle_idx] / 0.1)]
                    a = ax[1] if separate else ax
                    a.scatter(gabaX, gabaY, marker="v", c=[iClr], s=120, alpha=0.5)
            else:
                if not separate:
                    ax.scatter(achX, achY, c="g", s=120, alpha=0.5)
                    ax.scatter(gabaX, gabaY, marker="v", c="m", s=120, alpha=0.5)
                else:
                    ax[0].scatter(achX, achY, c="g", s=120, alpha=0.5)
                    ax[1].scatter(gabaX, gabaY, marker="v", c="m", s=120, alpha=0.5)

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
        x_px, y_px = dsgc.shape

        # offset whitespace and convert um locations to px
        syn_xs = np.array(self.syn_locs["X"]) * px_per_um + x_off
        syn_ys = np.array(self.syn_locs["Y"]) * px_per_um + y_off
        ach_xs = np.array(self.bp_locs["E"]["X"]) * px_per_um + x_off
        ach_ys = np.array(self.bp_locs["E"]["Y"]) * px_per_um + y_off
        gaba_xs = np.array(self.bp_locs["I"]["X"]) * px_per_um + x_off
        gaba_ys = np.array(self.bp_locs["I"]["Y"]) * px_per_um + y_off

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


def gaba_coverage_testing():
    """Resulting plot shows the relationship of the set gaba coverage value and
    the actual proportion of gaba containing synapses that would result. This calculation
    corresponds to the one used in `theta_picker_uniform` above. The relationship here
    is used to set the multiplier needed to allow gaba_coverage 0. -> 1.0 actually result
    in the desired gaba probability.

    NOTE:
    There is a hard switch from a slope of ~.5 from x = 0 to ~.9 to a slope of ~0.011
    from x ~.9 to 49.

    The inflection point `pt`, initial slope m0 and final slope m1 are tested to
    confirm that they enable approximate scaling.
    """

    def actual_coverage(gaba_coverage, reps=100):
        p = 0
        n = 1
        rand = h.Random(1)

        pt = 0.45
        m0 = 2
        m1 = 82
        if gaba_coverage > pt:
            g = (m0 * pt) + max(0, gaba_coverage - pt) * m1
        else:
            g = gaba_coverage * m0

        gaba_here = []
        for i in range(reps):
            base = rand.uniform(-180, 180)
            gaba_prob = p + (n - p) * (1 - 0.98 / (1 + np.exp(np.abs(base) - 96) / 25))
            gaba_prob *= g
            gaba_here.append(rand.uniform(0, 1) < gaba_prob)

        return np.sum(gaba_here) / reps

    x = np.concatenate([np.arange(50000) * 0.00002, (1 + np.arange(4700) * 0.01)])
    y = np.array([actual_coverage(g) for g in x])

    plt.plot(x, y)
    plt.xlabel("GABA Coverage Value")
    plt.ylabel("Actual Probability")

    plt.show()

    return x, y


def exp_rise(x, m, tau):
    return m * (1 - np.exp(-x / tau))


if __name__ == "__main__":
    gaba_coverage_testing()
