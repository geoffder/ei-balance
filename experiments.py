from neuron import h, gui

# science/math libraries
import numpy as np
import matplotlib.pyplot as plt

# general libraries
import h5py as h5
import json

# local imports
from modelUtils import find_spikes


class Rig:
    def __init__(self, model, data_path=""):
        self.model = model
        self.data_path = data_path
        self.model.soma.push()

    def place_electrodes(self):
        self.soma_rec = h.Vector()
        self.soma_rec.record(self.model.soma(0.5)._ref_v)
        self.soma_data = {"Vm": [], "area": [], "thresh_count": []}

        step = self.model.seg_step
        self.recs = {"Vm": [], "iCa": [], "g": {t: [] for t in ["E", "I"]}}
        self.dend_data = {
            "Vm": [], "iCa": [], "g": {t: [] for t in ["E", "I"]}
        }

        for n, dend in enumerate(self.model.all_dends):
            for i in range(self.model.rec_per_sec):
                self.recs["Vm"].append(h.Vector())
                self.recs["Vm"][-1].record(dend(i * step)._ref_v)

                self.recs["iCa"].append(h.Vector())
                self.recs["iCa"][-1].record(dend(i * step)._ref_ica)

        for n in range(self.model.n_syn):
            for t in self.recs["g"].keys():
                self.recs["g"][t].append(h.Vector())
                self.recs["g"][t][-1].record(
                    self.model.syns[t]["syn"][n]._ref_g)

    def dump_recordings(self):
        vm, area, count = self.measure_response(self.soma_rec)
        self.soma_data["Vm"].append(np.round(vm, decimals=3))
        self.soma_data["area"].append(np.round(area, decimals=3))
        self.soma_data["thresh_count"].append(count)

        for r in ["Vm", "iCa"]:
            if self.model.downsample[r] < 1:
                for rec in self.recs[r]:
                    rec.resample(rec, self.model.downsample[r])

            self.dend_data[r].append(np.round(self.recs[r], decimals=6))

        for t in self.recs["g"].keys():
            if self.model.downsample["g"] < 1:
                for rec in self.recs["g"][t]:
                    rec.resample(rec, self.model.downsample["g"])

            self.dend_data["g"][t].append(
                np.round(self.recs["g"][t], decimals=6))

    def clear_recordings(self):
        self.soma_rec.resize(0)

        for r in ["Vm", "iCa"]:
            for i in range(len(self.recs["Vm"])):
                self.recs[r][i].resize(0)

        for trans in self.recs["g"].values():
            for rec in trans:
                rec.resize(0)

    def summary(self, n_trials, plot=True):
        rads = self.model.dir_rads

        spikes = np.array(self.soma_data["thresh_count"]).reshape(n_trials, -1)
        metrics = {"DSis": [], "thetas": []}

        for i in range(n_trials):
            dsi, theta = self.calc_DS(rads, spikes[i])
            metrics["DSis"].append(dsi)
            metrics["thetas"].append(theta)

        metrics["DSis"] = np.round(metrics["DSis"], decimals=3)
        metrics["thetas"] = np.round(metrics["thetas"], decimals=1)

        metrics["spikes"] = spikes
        metrics["total_spikes"] = spikes.sum(axis=0)
        avg_dsi, avg_theta = self.calc_DS(rads, metrics["total_spikes"])
        metrics["avg_DSi"] = np.round(avg_dsi, decimals=3)
        metrics["avg_theta"] = np.round(avg_theta, decimals=2)
        metrics["DSis_sdev"] = np.std(metrics["DSis"]).round(decimals=2)
        metrics["thetas_sdev"] = np.std(metrics["thetas"]).round(decimals=2)
        metrics["num_GABA"] = np.sum(self.model.sac_net.gaba_here)

        self.print_results(metrics)
        if plot:
            self.plot_results(metrics)

        return metrics

    def print_results(self, metrics):
        rhos = (self.model.space_rho, self.model.time_rho)
        print("spRho: %.2f\ttmRho: %.2f" % rhos)
        if self.model.sac_mode:
            vars = self.model.sac_theta_vars
            print(
                "sacRho: %.2f\ttheta vars: E=%d, I=%d"
                % (self.model.sac_rho, vars["E"], vars["I"])
            )

        for k, v in metrics.items():
            print(k + ":", v)

        print("")

    def plot_results(self, metrics):
        fig1 = self.polar_plot(self.model.dir_labels, metrics)
        return fig1

    def run(self, stim, locked_synapses=None):
        """Initialize model, set synapse onset and release numbers, update
        membrane noise seeds and run the model. Calculate somatic response and
        return to calling function."""
        h.init()

        if stim["type"] == "flash":
            self.model.flash_onsets()
        elif stim["type"] == "bar":
            self.model.bar_onsets(stim, locked_synapses)

        self.model.set_failures(stim, locked_synapses)

        self.model.update_noise()

        self.clear_recordings()
        h.run()
        self.dump_recordings()

    def dir_run(
            self,
            n_trials=10,
            locked_synapses=None,
            rhos=None,
            prefix="",
            plot_summary=True
    ):
        """Run model through 8 directions for a number of trials and save the
        data. Offets and probabilities of release for inhibition are updated
        here before calling run() to execute the model.
        """
        self.place_electrodes()

        n_dirs = len(self.model.dirs)
        stim = {"type": "bar", "dir": 0}
        params = self.model.get_params_dict()  # for logging

        if rhos is not None:
            stim["rhos"] = rhos
            params["space_rho"] = rhos.get("space", 0)
            params["time_rho"] = rhos.get("time", 0)

        for j in range(n_trials):
            print("trial %d..." % j, end=" ", flush=True)

            for i in range(n_dirs):
                print("%d" % self.model.dir_labels[i], end=" ", flush=True)

                stim["dir"] = i
                self.run(stim, locked_synapses=locked_synapses)

            print("")  # next line

        metrics = self.summary(n_trials, plot=plot_summary)

        all_data = {
            "params": json.dumps(params),
            "metrics": metrics,
            "soma": {
                k: self.stack_trials(n_trials, n_dirs, v)
                for k, v in self.soma_data.items()
            },
            "dendrites": {
                "locs": self.model.get_recording_locations(),
                "Vm": self.stack_trials(
                    n_trials, n_dirs, self.dend_data["Vm"]),
                "iCa": self.stack_trials(
                    n_trials, n_dirs, self.dend_data["iCa"]),
                "g": {
                    k: self.stack_trials(n_trials, n_dirs, v)
                    for k, v in self.dend_data["g"].items()
                },
            },
        }

        if self.model.sac_net is not None:
            all_data["sac_net"] = self.model.sac_net.get_wiring_dict()

        self.pack_hdf(self.data_path + prefix + "dir_run", all_data)

        return metrics

    def rho_dir_run(self, n_trials=10, step=.1, seed_reset=True):
        """Repeat dir_run() experiments over a range of correlation (rho)
        values. First with spatial only, then temporal, then spatio-temporal
        correlations between excitation and inhibition.
        """
        rhos = {"space": 0, "time": 0}

        for dimensions in [["space"], ["time"], ["space", "time"]]:
            for i in range(int(1 / step)):
                if seed_reset:
                    self.model.seed = 0
                    self.model.nz_seed = 0

                for dim in dimensions:
                    rhos[dim] = i * step

                prefix = "rho_sp%.2f_tm%.2f_" % (rhos["space"], rhos["time"])
                self.dir_run(
                    n_trials, rhos=rhos, prefix=prefix, plot_summary=False
                )

    def offset_run(self, n_trials=10):
        offset_conds = {
            "control": {},  # no change
            "no_offset": {
                trans: {pn + "_offset": 0 for pn in ["pref", "null"]}
                for trans in ["E", "I"]
            },
            "symmetric": {
                "E": {pn + "_offset": -50 for pn in ["pref", "null"]},
                "I": {pn + "_offset": -55 for pn in ["pref", "null"]},
            },
        }

        gaba_prefix = "dir_gaba"
        for non_dir_gaba in range(2):
            self.model.seed = 0
            self.model.nz_seed = 0
            if non_dir_gaba:
                self.model.synprops["I"]["pref_prob"] = (
                    self.model.synprops["I"]["null_prob"])
                gaba_prefix = "non_dir_gaba"

            for cond, offsets in offset_conds.items():
                prefix = "%s_%s_" % (cond, gaba_prefix)
                self.model.update_params({"synprops": offsets})
                self.dir_run(n_trials, prefix=prefix, plot_summary=False)

    def sac_net_run(self, n_nets=3, n_trials=3, rho_steps=[0, .9]):
        """"""
        base_seed = self.model.sac_initial_seed
        metrics = {}
        for rho in rho_steps:
            initial_seed = base_seed
            self.model.seed = 0
            self.model.nz_seed = 0
            metrics[str(rho)] = []
            for i in range(n_nets):
                initial_seed += 1000
                prefix = "sac_rho%.2f_init_seed%i_" % (rho, initial_seed)
                self.model.build_sac_net(rho=rho, initial_seed=initial_seed)
                metrics[str(rho)].append(
                    self.dir_run(n_trials, prefix=prefix, plot_summary=False)
                )

        for rho, l in metrics.items():
            dsi = np.mean([m["avg_DSi"] for m in l])
            print("DSi @ sac_rho=%s: %.3f" % (rho, dsi))
        print("")

    def vc_dir_run(self, n_trials=10, simultaneous=True, prefix=""):
        """Similar to dir_run(), but running in voltage-clamp mode to record
        current at the soma. All other inputs are blocked when recording a
        particular syanptic input. Start script with vcPas=1 to block
        membrane channels prior to voltage-clamp experiments.
        """
        print("Voltage-Clamp Directional Run. Trials: %d" % n_trials)

        n_dirs = len(self.model.dirs)
        stim = {"type": "bar", "dir": 0}
        params = self.model.get_params_dict()  # for logging

        # create voltage clamp
        h("objref VC")
        VC = h.VC
        VC = h.SEClamp(0.5)

        # hold target voltage for entire duration
        VC.dur1 = h.tstop
        VC.dur2 = 0
        VC.dur3 = 0

        rec = h.Vector()
        rec.record(VC._ref_i)

        conditions = {
            "E": {"trans": ["E", "AMPA"], "holding": -60},
            "ACH": {"trans": ["E"], "holding": -60},
            "AMPA": {"trans": ["AMPA"], "holding": -60},
            "GABA": {"trans": ["I"], "holding": 0},
        }

        rec_data = {k: [] for k in conditions.keys()}

        for cond, settings in conditions.items():
            print("Condition: %s" % cond)

            if simultaneous:
                self.model.seed = 0
                self.model.nz_seed = 0

            VC.amp1 = settings["holding"]

            for t, ps in self.model.synprops.items():
                weight = ps["weight"] if t in settings["trans"] else 0
                for netcon in self.model.syns[t]["con"]:
                    for quanta in netcon:
                        quanta.weight[0] = weight

            for j in range(n_trials):
                print("trial %d..." % j, end=" ", flush=True)

                for i in range(n_dirs):
                    print("%d" % self.model.dir_labels[i], end=" ", flush=True)

                    stim["dir"] = i

                    h.init()

                    self.model.bar_onsets(stim)
                    self.model.set_failures(stim)
                    self.model.update_noise()

                    rec.resize(0)
                    h.run()
                    rec_data[cond].append(np.round(rec, decimals=5))

                print("")  # next line

        # reset all synaptic connection weights
        for cond, settings in conditions.items():
            for t, ps in self.model.synprops.items():
                for netcon in self.model.syns[t]["con"]:
                    for quanta in netcon:
                        quanta.weight[0] = ps["weight"]

        all_data = {
            "params": json.dumps(params),
            "soma": {
                k: self.stack_trials(n_trials, n_dirs, rec_data[k])
                for k in conditions.keys()
            }
        }

        self.pack_hdf(self.data_path + prefix + "vc_dir_run", all_data)

    def mini_test(self, idx, starts):
        """Turn off all synapses but for one and test. Input dict in form of
        {transmitter: time} indicates what should be on and when the release
        should occur. Use GUI to observe.
        """
        for t in self.model.synprops.keys():
            for stim in self.model.syns[t]["stim"]:
                for quanta in stim:
                    quanta.number = 0

        for t, time in starts.items():
            self.model.syns[t]["stim"][idx][0].number = 1
            self.model.syns[t]["stim"][idx][0].start = time

        h.run()

    def sac_angle_distribution(self, n_nets=100, bins=[8, 12, 16]):
        """Plot SAC dendrite angle distribution histograms, aggregating over a
        number of generated networks to get a more accurate estimate."""
        seed = self.model.sac_initial_seed
        iThetas, eThetas = [], []
        total_gaba = 0

        for _ in range(n_nets):
            seed += 1000
            self.model.build_sac_net(initial_seed=seed)
            iThetas.append(
                np.array(self.model.sac_net.thetas["I"]
                )[np.array(self.model.sac_net.gaba_here)]
            )
            eThetas.append(self.model.sac_net.thetas["E"])
            total_gaba += np.sum(self.model.sac_net.gaba_here)

        iThetas = np.concatenate(iThetas).flatten()
        eThetas = np.array(eThetas).flatten()
        print("Average GABA synapse count: %.2f" % (total_gaba / n_nets))

        fig, axes = plt.subplots(1, len(bins))

        for j, numBins in enumerate(bins):
            binAx = [i * 360 / numBins for i in range(numBins)]
            axes[j].hist(eThetas, bins=binAx, color="r", alpha=0.5)
            axes[j].hist(iThetas, bins=binAx, color="c", alpha=0.5)
            axes[j].set_title("Bin Size: %.1f" % (360 / numBins))
            axes[j].set_xlabel("SAC Dendrite Angle")
            axes[j].set_yticks([])

        return fig

    def synaptic_density(self, all_tree=False):
        first = 0 if all_tree else self.model.first_order
        dend_lens = [
            d.L for order in self.model.order_list[first:] for d in order
        ]
        density = len(self.model.syns["X"]) / np.sum(dend_lens)
        print(
            "There are %.4f synapses per micron (for %s dendrites)."
            % (density, "all" if all_tree else "synaptic")
        )

    @staticmethod
    def measure_response(vm_rec, threshold=20):
        vm = np.array(vm_rec)
        psp = vm + 61.3
        area = sum(psp[70:]) / len(psp[70:])
        thresh_count, _ = find_spikes(vm, thresh=threshold)
        return vm, area, thresh_count

    @staticmethod
    def calc_DS(dirs, response):
        xpts = np.multiply(response, np.cos(dirs))
        ypts = np.multiply(response, np.sin(dirs))
        xsum = np.sum(xpts)
        ysum = np.sum(ypts)
        DSi = np.sqrt(xsum ** 2 + ysum ** 2) / np.sum(response)
        theta = np.arctan2(ysum, xsum) * 180 / np.pi

        return DSi, theta

    @staticmethod
    def polar_plot(dirs, metrics):
        # resort directions and make circular for polar axes
        circ_vals = metrics["spikes"].T[np.array(dirs).argsort()]
        circ_vals = np.concatenate(
            [circ_vals, circ_vals[0, :].reshape(1, -1)], axis=0
        )
        circle = np.radians([0, 45, 90, 135, 180, 225, 270, 315, 0])

        peak = np.max(circ_vals)  # to set axis max
        avg_theta = np.radians(metrics["avg_theta"])
        avg_DSi = metrics["avg_DSi"]
        thetas = np.radians(metrics["thetas"])
        DSis = np.array(metrics["DSis"])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")

        # plot trials lighter
        ax.plot(circle, circ_vals, color=".75")
        ax.plot(
            [thetas, thetas], [np.zeros_like(DSis), DSis * peak], color=".75"
        )

        # plot avg darker
        ax.plot(circle, np.mean(circ_vals, axis=1), color=".0", linewidth=2)
        ax.plot(
            [avg_theta, avg_theta], [0.0, avg_DSi * peak],
            color=".0", linewidth=2
        )

        # misc settings
        ax.set_rlabel_position(-22.5)  # labels away from line
        ax.set_rmax(peak)
        ax.set_rticks([peak])
        ax.set_thetagrids([0, 90, 180, 270])

        plt.show()

        return fig

    @staticmethod
    def stack_trials(n_trials, n_dirs, data_list):
        """Stack a list of run recordings [ndarrays of shape (recs, samples)]
        into a single ndarray of shape (trials, directions, recs, samples).
        """
        stack = np.stack(data_list, axis=0)
        return stack.reshape(n_trials, n_dirs, *stack.shape[1:])

    @staticmethod
    def pack_hdf(pth, data_dict):
        """Takes data organized in a python dict, and creates an hdf5 with the
        same structure."""
        def rec(data, grp):
            for k, v in data.items():
                if type(v) is dict:
                    rec(v, grp.create_group(k))
                else:
                    grp.create_dataset(k, data=v)

        with h5.File(pth + ".h5", "w") as pckg:
            rec(data_dict, pckg)

    @staticmethod
    def unpack_hdf(group):
        """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
        return {
            k: v[()] if type(v) is h5._hl.dataset.Dataset else unpack_hdf(v)
            for k, v in group.items()
        }
