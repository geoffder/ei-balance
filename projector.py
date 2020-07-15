from neuron import h, gui

# science/math libraries
import numpy as np
import scipy.stats as st  # for probabilistic distributions

# general libraries
import platform

# local imports
from modelUtils import (
    build_stim,
    find_origin,
    rotate,
    merge,
)
import balance_configs as configs

"""TODO: Remove need for cell to be linked to and accessed directly from here.
-- instead, pass in the locations that sweep / etc need.
-- return times / success for each synapse, so that setting methods within the
   cell class can be called.
"""

class Projector:
    def __init__(self, cell, params=None):
        self.cell = cell
        self.set_default_params()

        # pre-calculations based on set properties
        self.hard_offsets = {
            trans: [
                self.dir_sigmoids["offset"](
                    d, p["pref_offset"], p["null_offset"]
                )
                for d in self.dir_labels
            ]
            for trans, p in self.cell.synprops.items()
        }

    def set_default_params(self):
        # hoc environment parameters
        self.tstop        = 250  # [ms]
        self.steps_per_ms = 10   # [10 = 10kHz]
        self.dt           = 0.1  # [ms, .1 = 10kHz]
        self.v_init       = -60
        self.celsius      = 36.9

        self.flash_mean     = 100  # mean onset time for flash stimulus
        self.flash_variance = 400  # variance of flash release onset
        self.jitter         = 60   # [ms] 10

        # light stimulus
        self.light_bar = {
            "start_time": 0,
            "speed":      1.0,   # 1#1.6 # speed of the stimulus bar (um/ms)
            "width":      250,   # width of the stimulus bar(um)
            "x_motion":   True,  # move bar in x, if not, move bar in y
            "x_start":    -40,   # start location (X axis) of the stim bar (um)
            "x_end":      200,   # end location (X axis)of the stimulus bar (um)
            "y_start":    25,    # start location (Y axis) of the stimulus bar (um)
            "y_end":      225,   # end location (Y axis) of the stimulus bar (um)
        }

        self.dir_labels = [225, 270, 315, 0, 45, 90, 135, 180]
        self.dir_rads   = np.radians(self.dir_labels)
        self.dirs       = [135, 90, 45, 0, 45, 90, 135, 180]
        self.dir_inds   = np.array(self.dir_labels).argsort()
        self.circle     = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0])

        # take direction, null, and preferred bounds and scale between
        self.dir_sigmoids = {
            "prob": lambda d, n, p: p
              + (n - p) * (1 - 0.98 / (1 + np.exp(d - 91) / 25)),
            "offset": lambda d, n, p: p
              + (n - p) * (1 - 0.98 / (1 + np.exp(d - 74.69) / 24.36)),
        }

        self.seed = 0

    def get_params_dict(self):
        skip = {
            "dir_sigmoids",
            "dir_rads",
            "dir_inds",
            "circle",
        }

        params = {
            k: v for k, v in self.__dict__.items()
            if k not in skip
        }

        return params

    def update_params(self, params):
        """Update self members with key-value pairs from supplied dict."""
        self.__dict__ = merge(self.__dict__, params)

    def flash_onsets(self):
        """
        Calculate onset times for each synapse randomly as though stimulus was
        a flash. Timing jitter is applied with pseudo-random number generators.

        TODO: Move to using returned onset times, strip out SETTING of
        syn.start times (move to cell class).
        """
        # distribution to pull mean onset times for each site from
        mOn = h.Random(self.seed)
        mOn.normal(self.flash_mean, self.flash_variance)
        self.seed += 1

        # store timings for return (fine to ignore return)
        onset_times = {trans: [] for trans in ["E", "I", "AMPA", "NMDA"]}

        rand_on = {trans: 0 for trans in ["E", "I", "AMPA", "NMDA"]}

        for s in range(self.cell.n_syn):
            onset = mOn.repick()

            for t in self.cell.synprops.keys():
                r = h.Random(self.seed)
                r.normal(0, 1)
                self.seed += 1
                rand_on[t] = r.repick()

            rand_on["E"] = rand_on["I"] * self.cell.time_rho + (
                rand_on["E"] * np.sqrt(1 - self.cell.time_rho ** 2)
            )

            for q in range(self.cell.max_quanta):
                if q:
                    # add variable delay til next quanta (if there is one)
                    quanta_delay = h.Random(self.seed)
                    quanta_delay.normal(
                        self.cell.quanta_inter, self.cell.quanta_inter_var
                    )
                    self.seed += 1
                    onset += quanta_delay.repick()

                for t, props in self.cell.synprops.items():
                    self.cell.syns[t]["stim"][s][q].start = (
                        onset + props["delay"] + rand_on[t] * props["var"]
                    )

                    onset_times[t].append(rand_on[t])

                if self.cell.NMDA_exc_lock:
                    self.cell.syns["NMDA"]["stim"][s][q].start = (
                        self.cell.syns["E"]["stim"][s][q].start
                    )

        return onset_times

    def bar_sweep(self, syn, dir_idx):
        """Return activation time for the given synapse based on the light bar
        config and the pre-synaptic setup (e.g. SAC network/hard-coded offsets)
        """
        bar = self.light_bar
        rotation = -self.dir_rads[dir_idx]
        x, y = rotate(
            self.cell.origin,
            self.cell.syns["X"][syn],
            self.cell.syns["Y"][syn],
            rotation
        )
        loc = {"x": x, "y": y}
        ax = "x" if bar["x_motion"] else "y"

        on_times = {}

        # distance to synapse divided by speed
        for t, props in self.cell.synprops.items():
            if t in ["E", "I"] and self.cell.sac_net is not None:
                sac_loc = self.cell.sac_net.get_syn_loc(t, syn, rotation)
                on_times[t] = (
                    bar["start_time"]
                    + (sac_loc[ax] - bar[ax + "_start"]) / bar["speed"]
                )
            else:
                offset = self.hard_offsets[t][dir_idx]
                on_times[t] = (
                    bar["start_time"]
                    + (loc[ax] + offset - bar[ax + "_start"]) / bar["speed"]
                )

        return on_times

    def bar_onsets(self, stim, locked_synapses=None):
        """
        Calculate onset times for each synapse based on when the simulated bar
        would be passing over their location, modified by spatial offsets.
        Timing jitter is applied using pseudo-random number generators.

        TODO: Move to using returned onset times, strip out SETTING of
        syn.start times (move to cell class).
        """
        sac = self.cell.sac_net

        time_rho = stim.get("rhos", {"time": self.cell.time_rho})["time"]

        # store timings for return (fine to ignore return)
        onset_times = {trans: [] for trans in ["E", "I", "AMPA", "NMDA"]}

        rand_on = {trans: 0 for trans in ["E", "I", "AMPA", "NMDA"]}

        for s in range(self.cell.n_syn):
            bar_times = self.bar_sweep(s, stim["dir"])
            if (sac is None
                    or not self.cell.sac_angle_rho_mode
                    or not sac.gaba_here[s]):
                syn_rho = time_rho
            else:
                # scale correlation of E and I by diff of their dend angles
                syn_rho = time_rho - time_rho * sac.deltas[s] / 180

            # shared jitter
            jit_rand = h.Random(self.seed)
            jit_rand.normal(0, 1)
            self.seed += 1
            jit = jit_rand.repick()
            bar_times = {
                k: v + jit * self.jitter
                for k, v in bar_times.items()
            }

            for t in self.cell.synprops.keys():
                rand_on[t] = h.Random(self.seed).normal(0, 1)
                self.seed += 1

            rand_on["E"] = rand_on["I"] * syn_rho + (
                rand_on["E"] * np.sqrt(1 - syn_rho ** 2)
            )

            for q in range(self.cell.max_quanta):
                if q:
                    # add variable delay til next quanta (if there is one)
                    quanta_rand = h.Random(self.seed)
                    quanta_rand.normal(
                        self.cell.quanta_inter, self.cell.quanta_inter_var
                    )
                    self.seed += 1
                    q_delay = quanta_rand.repick()
                    bar_times = {k: v + q_delay for k, v in bar_times.items()}

                for t, props in self.cell.synprops.items():
                    self.cell.syns[t]["stim"][s][q].start = (
                        bar_times[t] + props["delay"]
                        + rand_on[t] * props["var"]
                    )

                    onset_times[t].append(rand_on[t])

                # TODO: Maybe remove this option since I don't think it's used
                if self.cell.NMDA_exc_lock:
                    self.cell.syns["NMDA"]["stim"][s][q].start = (
                        self.cell.syns["E"]["stim"][s][q].start
                    )

        if locked_synapses is not None:
            # reset activation time to coincide with bar (no noise)
            for s in locked_synapses["nums"]:
                lock_times = self.bar_sweep(s, stim["dir"])
                for t in ["E", "I"]:
                    self.cell.syns[t]["stim"][s][0].start = lock_times[t]

        return onset_times

    def set_failures(self, stim, locked_synapses=None):
        """
        Determine number of quantal activations of each synapse occur on a
        trial. Psuedo-random numbers generated for each synapse are compared
        against thresholds set by probability of release to determine if the
        "pre-synapse" succeeds or fails to release neurotransmitter.

        TODO: Move to using returned onset times, strip out SETTING of
        syn.start times (move to cell class).
        """
        sac = self.cell.sac_net

        space_rho = stim.get("rhos", {"space": self.cell.space_rho})["space"]

        # numbers above can result in NaNs
        rho = 0.986 if space_rho > 0.986 else space_rho

        # calculate input rho required to achieve the desired output rho
        # exponential fit: y = y0 + A * exp(-invTau * x)
        # y0 = 1.0461; A = -0.93514; invTau = 3.0506
        rho = 1.0461 - 0.93514 * np.exp(-3.0506 * rho)

        rand_succ = {}
        for t in self.cell.synprops.keys():
            rand_succ[t] = h.Random(self.seed)
            rand_succ[t].normal(0, 1)
            self.seed += 1

        picks = {
            t: np.array([rand_succ[t].repick() for i in range(self.cell.n_syn)])
            for t in self.cell.synprops.keys()
        }

        # correlate synaptic variance of ACH with GABA
        if sac is None or not self.cell.sac_angle_rho_mode:
            picks["E"] = picks["I"] * rho + (
                picks["E"] * np.sqrt(1 - rho ** 2))
        else:
            # scale correlation of E and I by prox of their dend angles
            for i in range(self.cell.n_syn):
                if sac.gaba_here[i]:
                    syn_rho = rho - rho * sac.deltas[i] / 180
                    picks["E"][i] = (
                        picks["I"][i] * syn_rho
                        + picks["E"][i] * np.sqrt(1 - syn_rho ** 2)
                    )

        probs = {}
        for t, props in self.cell.synprops.items():
            if stim["type"] == "flash":
                probs[t] = np.full(self.cell.n_syn, props["prob"])
            elif sac is not None and t in ["E", "I"]:
                probs[t] = sac.probs[t][:, stim["dir"]]
            else:
                # calculate probability of release
                p = self.dir_sigmoids["prob"](
                    self.dirs[stim["dir"]],
                    props["pref_prob"],
                    props["null_prob"]
                )
                probs[t] = np.full(self.cell.n_syn, p)

        sdevs = {k: np.std(v) for k, v in picks.items()}
        successes = {}

        for t in picks.keys():
            # determine release bool for each possible quanta
            q_probs = np.array([
                probs[t] * (self.cell.quanta_Pr_decay ** q)
                for q in range(self.cell.max_quanta)
            ])
            left = st.norm.ppf((1 - q_probs) / 2.0) * sdevs[t]
            right = st.norm.ppf(1 - (1 - q_probs) / 2.0) * sdevs[t]
            successes[t] = (left < picks[t]) * (picks[t] < right)

            for i in range(self.cell.n_syn):
                for q in range(self.cell.max_quanta):
                    if successes[t][q][i]:
                        self.cell.syns[t]["stim"][i][q].number = 1
                    else:
                        self.cell.syns[t]["stim"][i][q].number = 0

        if locked_synapses is not None:
            for s in locked_synapses["nums"]:
                self.cell.syns[t]["con"][s][0].weight[0] = (
                    self.cell.synprops["I"]["weight"] * 10)
                for t in ["E", "I"]:
                    self.cell.syns[t]["stim"][s][0].number = locked_synapses[t]

        pearson = st.pearsonr(
            successes["E"].flatten(), successes["I"].flatten()
        )
        return pearson[0]

        
