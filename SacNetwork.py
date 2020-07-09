from neuron import h, gui

import numpy as np
import matplotlib.pyplot as plt

import json

from modelUtils import rotate


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
                    prs[-1] = np.abs(
                        prs[-1] - 360) if prs[-1] > 180 else prs[-1]
                    prs[-1] = pref + (null - pref) * (
                        1 - 0.98 / (1 + np.exp((prs[-1] - 91) / 25))
                    )

                self.probs[t].append(prs)

        self.origin = self.find_origin()
        self.gaba_here = np.array(self.gaba_here)
        
        for t in ["E", "I"]:
            self.probs[t] = np.nan_to_num(self.probs[t])
            self.thetas[t] = np.array(self.thetas[t])
            self.bp_locs[t]["X"] = np.array(self.bp_locs[t]["X"])
            self.bp_locs[t]["Y"] = np.array(self.bp_locs[t]["Y"])

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
            theta = self.wrap_360(
                pick[t] * self.theta_vars[t] + shared + base + self.cell_pref)

            self.thetas[t].append(theta)

    def theta_picker_cardinal(self):
        base = (self.rand.uniform(0, 1) // .25) * 90

        if base == 180:
            gaba_prob = 2 * self.gaba_coverage * 1
        elif base == 90 or base == 270:
            gaba_prob = 2 * self.gaba_coverage * .25
        else:
            gaba_prob = 2 * self.gaba_coverage * 0

        self.gaba_here.append(self.rand.uniform(0, 1) < gaba_prob)
        
        # pseudo-random numbers for angle determination (outer portion)
        pick = self.get_outer_picks(self.gaba_here[-1])
            
        for t in ["E", "I"]:
            if t == "I" and not self.gaba_here[-1]:
                self.thetas["I"].append(np.NaN)
                continue

            theta = self.wrap_360(pick[t] * self.theta_vars[t] + base)

            self.thetas[t].append(theta)
        
    def theta_picker_uniform(self):
        p = 0
        n = 1
        base = self.rand.uniform(-180, 180)
        
        gaba_prob = p + (n - p) * (
            # 1 - 0.98 / (1 + np.exp(np.abs(base) - 91) / 25))
            1 - 0.98 / (1 + np.exp(np.abs(base) - 96) / 25))
        # gaba_prob *= 2 * self.gaba_coverage  # NOTE: Approximate, fix if used...
        # NOTE: More restrictive sigmoid decr GABA coverage, so bumped up...
        gaba_prob *= 6 * self.gaba_coverage
        self.gaba_here.append(self.rand.uniform(0, 1) < gaba_prob)
        
        pick = self.get_outer_picks(self.gaba_here[-1])
            
        for t in ["E", "I"]:
            if t == "I" and not self.gaba_here[-1]:
                self.thetas["I"].append(np.NaN)
                continue

            theta = self.wrap_360(
                pick[t] * self.theta_vars[t] + base + self.cell_pref)

            self.thetas[t].append(theta)
            
    def theta_picker_uniform_post(self):
        p = 0.05
        n = 1
        base = self.rand.uniform(-180, 180)
        
        # pseudo-random numbers for angle determination (outer portion)
        pick = self.get_outer_picks(True)

        for t in ["E", "I"]:
            theta = self.wrap_360(
                pick[t] * self.theta_vars[t] + base + self.cell_pref)

            if t == "I":
                d = np.abs(theta - (theta // 180) * 360)
                gaba_prob = p + (n - p) * (
                    1 - 0.98 / (1 + np.exp(d - 120) / 25))
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
            pick["E"] = (pick["I"] * self.rho
                         + pick["E"] * np.sqrt(1 - self.rho ** 2))
        return pick

    def find_origin(self):
        allX = np.concatenate(
            [self.bp_locs["E"]["X"], self.bp_locs["I"]["X"]], axis=0
        )
        allX = allX[~np.isnan(allX)]
        leftX, rightX = np.min(allX), np.max(allX)

        allY = np.concatenate(
            [self.bp_locs["E"]["Y"], self.bp_locs["I"]["Y"]], axis=0
        )
        allY = allY[~np.isnan(allY)]
        botY, topY = np.min(allY), np.max(allY)

        return (leftX + (rightX - leftX) / 2, botY + (topY - botY) / 2)

    def get_syn_loc(self, trans, num, rotation):
        x, y = rotate(
            self.origin,
            self.bp_locs[trans]["X"][num],
            self.bp_locs[trans]["Y"][num],
            rotation
        )
        return {"x": x, "y": y}

    def plot_dends(self, stim_angle=None):
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
            ax.plot([synX, achX], [synY, achY], c="g", alpha=0.5)
            ax.plot([synX, gabaX], [synY, gabaY], c="m", alpha=0.5)

            if stim_angle is not None:
                angle_idx = np.argwhere(
                    self.dir_labels == np.array(stim_angle))[0][0]
                colors = [plt.get_cmap("jet")(1.0 * i / 10) for i in range(10)]
                eClr = colors[int(ePr[angle_idx] / 0.1)]
                ax.scatter(achX, achY, c=[eClr], s=120, alpha=0.5)

                if not np.isnan(gabaX):
                    iClr = colors[int(iPr[angle_idx] / 0.1)]
                    ax.scatter(
                        gabaX, gabaY, marker="v", c=[iClr], s=120, alpha=0.5
                    )
            else:
                ax.scatter(achX, achY, c="g", s=120, alpha=0.5)
                ax.scatter(gabaX, gabaY, marker="v", c="m", s=120, alpha=0.5)

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
        param_keys = {
            "dir_pr", "rho", "theta_vars", "gaba_coverage", "offset", "seed"
        }

        net = {
            "params": json.dumps({
                k: v for k, v in self.__dict__.items()
                if k in param_keys
            }),
            "wiring": {
                k: v for k, v in self.__dict__.items()
                if k in {"gaba_here", "bp_locs", "thetas", "probs"}
            },
        }

        return net

    @staticmethod
    def wrap_360(theta):
        """Wrap degrees from -180 -> 180 scale and super 360 to 0 -> 360 scale.
        """
        theta %= 360
        if theta < 0:
            return theta + 360
        else:
            return theta
        
