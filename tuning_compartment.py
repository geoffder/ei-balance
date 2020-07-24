from neuron import h

# science/math libraries
import numpy as np
import matplotlib.pyplot as plt

# general libraries
import platform

# local imports
from modelUtils import (
    windows_gui_fix,
    nrn_objref,
    nrn_section,
    build_stim,
    find_origin,
    rotate,
    merge,
    wrap_180,
)
from SacNetwork import SacNetwork
from experiments import Rig


"""Toy model for investigating the impact on tuning that differing E/I input
dendrite angles would have.

-- 8 directional drifting edge stimulus
-- fixed angle inhibitory dendrite
-- variable angle excitatory dendrite (testing around the clock)
-> how does the post-synaptic directional tuning change with E/I theta delta?
"""


class TuningToy:
    def __init__(self, seed=0):
        self.origin = {"X": 100, "Y": 100}
        self.offset = 30
        self.dir_pr = {
            "E": {"pref": 0.95, "null": 0.05},
            "I": {"pref": 0.95, "null": 0.05},
        }
        self.syn_timing = {
            "E": {"var": 15, "delay": 5},
            "I": {"var": 15, "delay": 0},
        }

        self.dir_labels = [225, 270, 315, 0, 45, 90, 135, 180]
        self.dir_rads   = np.radians(self.dir_labels)
        self.dirs       = [135, 90, 45, 0, 45, 90, 135, 180]
        self.dir_inds   = np.array(self.dir_labels).argsort()
        self.circle     = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0])

        self.light_bar = {
            "speed": 1., "x_motion": True, "x_start": -60, "y_start": -70
        }

        self.config_soma()
        self.create_synapse()

        self.seed = seed
        self.nz_seed = 0
        self.rand = h.Random(seed)

    def config_soma(self):
        """Build and set membrane properties of soma compartment"""
        self.soma = nrn_section("soma")

        self.soma.L    = 10  # [um]
        self.soma.diam = 10  # [um]
        self.soma.nseg = 1
        self.soma.Ra   = 100

        self.soma.insert("HHst")
        self.soma.gnabar_HHst = 0.013      # [S/cm2]
        self.soma.gkbar_HHst  = 0.035      # [S/cm2]
        self.soma.gkmbar_HHst = 0.003      # [S/cm2]
        self.soma.gleak_HHst  = 0.0001667  # [S/cm2]
        self.soma.eleak_HHst  = -60.0      # [mV]
        self.soma.NF_HHst     = 0.25

    def config_stimulus(self):
        # light stimulus
        self.light_bar = {
            "start_time": 0,
            "speed": 1.0,      # speed of the stimulus bar (um/ms)
            "width": 250,      # width of the stimulus bar(um)
            "x_motion": True,  # move bar in x, if not, move bar in y
            "x_start": -40,    # start location (X axis) of the stim bar (um)
            "x_end": 200,      # end location (X axis)of the stimulus bar (um)
            "y_start": 25,     # start location (Y axis) of the stimulus bar (um)
            "y_end": 225,      # end location (Y axis) of the stimulus bar (um)
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

    def create_synapse(self):
        # complete synapses are made up of a NetStim, Syn, and NetCon
        self.syns = {
            "E": {"stim": build_stim(), "syn": h.Exp2Syn(0.5)},
            "I": {"stim": build_stim(), "syn": h.Exp2Syn(0.5)},
        }

        self.syns["E"]["syn"].tau1 = 0.1
        self.syns["E"]["syn"].tau2 = 4.0
        self.syns["E"]["syn"].e    = 0

        self.syns["I"]["syn"].tau1 = 0.5
        self.syns["I"]["syn"].tau2 = 12.0
        self.syns["I"]["syn"].e    = -60.0

        self.syns["E"]["con"] = h.NetCon(
            self.syns["E"]["stim"], self.syns["E"]["syn"], 0, 0, .001
        )
        self.syns["I"]["con"] = h.NetCon(
            self.syns["I"]["stim"], self.syns["I"]["syn"], 0, 0, .003
        )

    def set_sacs(self, thetas={"E": 0, "I": 0}):
        self.thetas = thetas

        self.bp_locs = {
            s: {
                "X": self.origin["X"] - self.offset * np.cos(np.deg2rad(t)),
                "Y": self.origin["Y"] - self.offset * np.sin(np.deg2rad(t)),
            } for s, t in thetas.items()
        }

        self.prs = {
            s: [
                pn["pref"] + (pn["null"] - pn["pref"]) * (
                    1 - 0.98 / (1 + np.exp((wrap_180(np.abs(t - d)) - 91) / 25))
                ) for d in self.dir_labels
            ] for (s, t), pn in zip(thetas.items(), self.dir_pr.values())
        }

        # theta difference used to scale down rho
        self.delta = wrap_180(np.abs(thetas["E"] - thetas["I"]))

    def rotate_sacs(self, rotation):
        rotated = {}
        for s, locs in self.bp_locs.items():
            x, y = rotate(self.origin, locs["X"], locs["Y"], rotation)
            rotated[s] = {"X": x, "Y": y}
        return rotated

    def bar_sweep(self, dir_idx):
        """Return activation time for the single synapse based on the light bar
        config and the bipolar locations on the presynaptic dendrites.
        """
        bar = self.light_bar
        ax = "x" if bar["x_motion"] else "y"
        locs = rotate_sacs(-self.dir_rads[dir_idx])

        on_times = {
            s: bar["start_time"] + (loc[ax] - bar[ax + "_start"] / bar["speed"])
            for s, loc in locs.items()
        }

        return on_times

    def bar_onsets(self, stim):
        """Calculate onset times for each synapse based on when the simulated
        bar would be passing over their location, modified by spatial offsets.
        Timing jitter is applied using pseudo-random number generators.
        """
        sac = self.sac_net

        time_rho = stim.get("rhos", {"time": self.time_rho})["time"]

        if (self.sac_angle_rho_mode):
            syn_rho = time_rho
        else:
            syn_rho = time_rho - time_rho * self.delta / 180

        # bare base onset with added shared jitter
        jit = self.rand.normal(0, 1)
        bar_times = {
            k: v + jit * self.jitter
            for k, v in self.bar_sweep(stim["dir"]).items()
        }

        rand_on = {t: self.rand.normal(0, 1) for t in self.bar_times.keys()}
        rand_on["E"] = rand_on["I"] * syn_rho + (
            rand_on["E"] * np.sqrt(1 - syn_rho ** 2)
        )

        for t, timing in self.syn_timing.items():
            self.syns[t]["stim"].start = (
                bar_times[t] + timing["delay"] + rand_on[t] * timing["var"]
            )

    def set_failures(self, stim):
        """Determine whether each synapse has a release event or not.
        Psuedo-random numbers generated for each synapse are compared against
        thresholds set by probability of release to determine if the
        "pre-synapse" succeeds or fails to release neurotransmitter.
        """
        space_rho = stim.get("rhos", {"space": self.space_rho})["space"]

        # numbers above can result in NaNs
        rho = 0.986 if space_rho > 0.986 else space_rho

        # calculate input rho required to achieve the desired output rho
        # exponential fit: y = y0 + A * exp(-invTau * x)
        # y0 = 1.0461; A = -0.93514; invTau = 3.0506
        rho = 1.0461 - 0.93514 * np.exp(-3.0506 * rho)

        picks = {t: self.rand.normal(0, 1) for t in self.syns.keys()}

        # correlate synaptic variance of ACH with GABA
        if not self.sac_angle_rho_mode:
            picks["E"] = picks["I"] * rho + (
                picks["E"] * np.sqrt(1 - rho ** 2))
        else:
            # scale correlation of E and I by prox of their dend angles
            syn_rho = rho - rho * self.delta / 180
            picks["E"] = (
                picks["I"] * syn_rho
                + picks["E"] * np.sqrt(1 - syn_rho ** 2)
            )

        for t in picks.keys():
            prob = self.prs[t][stim["dir"]]
            sdev = np.std(self.picks[t])

            left = st.norm.ppf((1 - prob) / 2.0) * sdev
            right = st.norm.ppf(1 - (1 - prob) / 2.0) * sdev
            success = (left < picks[t]) * (picks[t] < right)

            self.syns[t]["stim"].number = int(success)

    def update_noise(self):
        self.soma.seed_HHst = self.nz_seed
        self.nz_seed += 1


class Runner:
    def __init__(self):
        self.model = TuningToy()

    def run(self, stim):
        """Initialize model, set synapse onset and release numbers, update
        membrane noise seeds and run the model. Calculate somatic response and
        return to calling function."""
        h.init()

        self.model.bar_onsets(stim)
        self.model.set_failures(stim)
        self.model.update_noise()

        self.clear_recordings()
        h.run()
        self.dump_recordings()

    def place_electrode(self):
        self.soma_rec = h.Vector()
        self.soma_rec.record(self.model.soma(0.5)._ref_v)
        self.soma_data = {"Vm": [], "area": [], "thresh_count": []}

    def dump_recordings(self):
        vm, area, count = Rig.measure_response(self.soma_rec)
        self.soma_data["Vm"].append(np.round(vm, decimals=3))
        self.soma_data["area"].append(np.round(area, decimals=3))
        self.soma_data["thresh_count"].append(count)

    def clear_recordings(self):
        self.soma_rec.resize(0)

    def summary(self, n_trials, plot=True):
        rads = self.dir_rads

        spikes = np.array(self.soma_data["thresh_count"]).reshape(n_trials, -1)
        metrics = {"DSis": [], "thetas": []}

        for i in range(n_trials):
            dsi, theta = Rig.calc_DS(rads, spikes[i])
            metrics["DSis"].append(dsi)
            metrics["thetas"].append(theta)

        metrics["DSis"] = np.round(metrics["DSis"], decimals=3)
        metrics["thetas"] = np.round(metrics["thetas"], decimals=1)

        metrics["spikes"] = spikes
        metrics["total_spikes"] = spikes.sum(axis=0)
        avg_dsi, avg_theta = Rig.calc_DS(rads, metrics["total_spikes"])
        metrics["avg_DSi"] = np.round(avg_dsi, decimals=3)
        metrics["avg_theta"] = np.round(avg_theta, decimals=2)
        metrics["DSis_sdev"] = np.std(metrics["DSis"]).round(decimals=2)
        metrics["thetas_sdev"] = np.std(metrics["thetas"]).round(decimals=2)

        self.print_results(metrics)
        if plot:
            self.plot_results(metrics)

        return metrics

    def print_results(self, metrics):
        for k, v in metrics.items():
            print(k + ":", v)

        print("")

    def plot_results(self, metrics):
        fig1 = Rig.polar_plot(self.dir_labels, metrics)
        return fig1


def set_hoc_params():
    """Set hoc NEURON environment model run parameters."""
    h.tstop = 100  # [ms]
    h.steps_per_ms = 10
    h.dt = 0.1  # [ms]
    h.v_init = -60  # [mV]
    h.celsius = 36.9


if __name__ == "__main__":
    if platform.system() == "Linux":
        basest = "/mnt/Data/NEURONoutput/"
    else:
        basest = "D:\\NEURONoutput\\"
        windows_gui_fix()

    toy = TuningToy()
    toy.set_sacs()
