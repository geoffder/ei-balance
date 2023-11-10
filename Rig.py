from neuron import h

# science/math libraries
import numpy as np
import matplotlib.pyplot as plt

# general libraries
import os
import h5py as h5
import json
from typing import List

# local imports
from modelUtils import find_spikes
from general_utils import clean_axes
from hdf_utils import pack_hdf


class Rig:
    def __init__(self, model, data_path=""):
        self.model = model
        self.data_path = data_path
        self.model.soma.push()
        self.initialize_handler = h.FInitializeHandler(self.model.init_synapses)

    def place_electrodes(self):
        self.soma_rec = h.Vector()
        self.soma_rec.record(self.model.soma(0.5)._ref_v)
        self.soma_data = {"Vm": [], "area": [], "thresh_count": []}

        step = self.model.seg_step
        self.recs = {"Vm": [], "iCa": [], "cai": [], "g": {t: [] for t in ["E", "I"]}}
        self.dend_data = {
            "Vm": [],
            "iCa": [],
            "cai": [],
            "g": {t: [] for t in ["E", "I"]},
        }

        if self.model.record_tree:
            for n, dend in enumerate(self.model.all_dends):
                for i in range(self.model.rec_per_sec):
                    self.recs["Vm"].append(h.Vector())
                    self.recs["Vm"][-1].record(dend(i * step)._ref_v)

                    self.recs["iCa"].append(h.Vector())
                    self.recs["iCa"][-1].record(dend(i * step)._ref_ica)

                    self.recs["cai"].append(h.Vector())
                    self.recs["cai"][-1].record(dend(i * step)._ref_cai)

            for n in range(self.model.n_syn):
                for t in self.recs["g"].keys():
                    self.recs["g"][t].append(h.Vector())
                    self.recs["g"][t][-1].record(self.model.syns[t]["syn"][n]._ref_g)

    def dump_recordings(self):
        vm, area, count = self.measure_response(self.soma_rec)
        self.soma_data["Vm"].append(np.round(vm, decimals=3))
        self.soma_data["area"].append(np.round(area, decimals=3))
        self.soma_data["thresh_count"].append(count)

        if self.model.record_tree:
            for r in ["Vm", "iCa", "cai"]:
                if self.model.downsample[r] < 1:
                    for rec in self.recs[r]:
                        rec.resample(rec, self.model.downsample[r])

                self.dend_data[r].append(np.round(self.recs[r], decimals=6))

            for t in self.recs["g"].keys():
                if self.model.downsample["g"] < 1:
                    for rec in self.recs["g"][t]:
                        rec.resample(rec, self.model.downsample["g"])

                self.dend_data["g"][t].append(np.round(self.recs["g"][t], decimals=6))

    def clear_recordings(self):
        self.soma_rec.resize(0)

        for r in ["Vm", "iCa", "cai"]:
            for i in range(len(self.recs["Vm"])):
                self.recs[r][i].resize(0)

        for trans in self.recs["g"].values():
            for rec in trans:
                rec.resize(0)

    def summary(self, n_trials, plot=True, quiet=False):
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
        if self.model.sac_mode:
            metrics["num_GABA"] = np.sum(self.model.sac_net.gaba_here)

        if not quiet:
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

    def plot_results(self, metrics, show_plot=True):
        fig1 = self.polar_plot(self.model.dir_labels, metrics, show_plot)
        return fig1

    def run(self, stim):
        """Initialize model, set synapse onset and release numbers, update
        membrane noise seeds and run the model. Calculate somatic response and
        return to calling function."""
        h.init()

        if stim["type"] == "flash":
            self.model.flash_onsets()
        elif stim["type"] == "bar":
            if self.model.poisson_mode:
                self.model.bar_poissons(stim)
            else:
                self.model.bar_onsets(stim)

        self.model.update_noise()

        self.clear_recordings()
        h.run()
        self.dump_recordings()
        self.model.clear_synapses()

    def dir_run(
        self, n_trials=10, rhos=None, save_name=None, plot_summary=True, quiet=False
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
            self.model.space_rho = params["space_rho"]
            self.model.time_rho = params["time_rho"]

        for j in range(n_trials):
            if not quiet:
                print("trial %d..." % j, end=" ", flush=True)

            for i in range(n_dirs):
                if not quiet:
                    print("%d" % self.model.dir_labels[i], end=" ", flush=True)

                stim["dir"] = i
                self.run(stim)

            if not quiet:
                print("")  # next line

        metrics = self.summary(n_trials, plot=plot_summary, quiet=quiet)

        all_data = {
            "params": params,
            "metrics": metrics,
            "soma": {
                k: self.stack_trials(n_trials, n_dirs, v)
                for k, v in self.soma_data.items()
            },
            "syn_locs": self.model.syn_locs,
        }
        if self.model.record_tree:
            all_data["dendrites"] = {
                "locs": self.model.get_recording_locations(),
                "Vm": self.stack_trials(n_trials, n_dirs, self.dend_data["Vm"]),
                "iCa": self.stack_trials(n_trials, n_dirs, self.dend_data["iCa"]),
                "cai": self.stack_trials(n_trials, n_dirs, self.dend_data["cai"]),
                "g": {
                    k: self.stack_trials(n_trials, n_dirs, v)
                    for k, v in self.dend_data["g"].items()
                },
            }

        if self.model.sac_net is not None:
            all_data["sac_net"] = self.model.sac_net.get_wiring_dict()

        if save_name is not None:
            pack_hdf(os.path.join(self.data_path, save_name), all_data)
            return metrics
        else:
            return all_data

    def rho_dir_run(
        self,
        n_trials=10,
        step=0.1,
        seed_reset=True,
        save_prefix=None,
        plot_summary=True,
        quiet=False,
    ):
        """Repeat dir_run() experiments over a range of correlation (rho)
        values. First with spatial only, then temporal, then spatio-temporal
        correlations between excitation and inhibition.
        """
        rhos = {"space": 0, "time": 0}

        for dimensions in [["space"], ["time"], ["space", "time"]]:
            for i in range(int(1 / step)):
                if seed_reset:
                    self.model.reset_rng()

                for dim in dimensions:
                    rhos[dim] = i * step

                if save_prefix is not None:
                    save_name = "rho_sp%.2f_tm%.2f_dir_run" % (
                        rhos["space"],
                        rhos["time"],
                    )
                else:
                    save_name = None

                self.dir_run(
                    n_trials,
                    rhos=rhos,
                    save_name=save_name,
                    plot_summary=plot_summary,
                    quiet=quiet,
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
                self.model.synprops["I"]["pref_prob"] = self.model.synprops["I"][
                    "null_prob"
                ]
                gaba_prefix = "non_dir_gaba"

            for cond, offsets in offset_conds.items():
                prefix = "%s_%s_" % (cond, gaba_prefix)
                self.model.update_params({"synprops": offsets})
                self.dir_run(n_trials, prefix=prefix, plot_summary=False)

    def sac_net_run(
        self, n_nets=3, n_trials=3, rho_steps=[0, 0.9], save_runs=True, quiet=False
    ):
        """"""
        metrics = {}
        for rho in rho_steps:
            self.model.nz_seed = 0
            metrics[str(rho)] = []
            for i in range(n_nets):
                name = "sac_rho%.2f_net%i_dir_run_" % (rho, i) if save_runs else None
                self.model.build_sac_net(rho=rho)
                metrics[str(rho)].append(
                    self.dir_run(n_trials, save_name=name, plot_summary=False)
                )

        if not quiet:
            for rho, l in metrics.items():
                dsi = np.mean([m["avg_DSi"] for m in l])
                print("DSi @ sac_rho=%s: %.3f" % (rho, dsi))
            print("")

    def gaba_coverage_run(
        self,
        n_nets=3,
        n_trials=3,
        rho_steps=[0.0, 0.9],
        cov_steps=[0.1, 0.15, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70],
    ):
        base_data_path = self.data_path[:]
        base_seed = self.model.sac_initial_seed
        for c in cov_steps:
            label = "coverage%i" % int(c * 100)
            self.data_path = os.path.join(base_data_path, label)
            os.makedirs(self.data_path, exist_ok=True)
            self.model.sac_gaba_coverage = c
            self.model.sac_initial_seed = base_seed
            self.sac_net_run(n_nets=n_nets, n_trials=n_trials, rho_steps=rho_steps)

        self.data_path = base_data_path[:]

    def gaba_titration_run(
        self,
        n_nets=3,
        n_trials=3,
        rho_steps=[0.0, 0.9],
        multi=[0.10, 0.15, 0.25, 0.50, 0.75, 1.0, 1.5],
    ):
        base_data_path = self.data_path[:]
        base_gaba_weight = self.model.synprops["I"]["weight"] * 1.0

        for m in multi:
            self.data_path = os.path.join(
                base_data_path, "gaba_scale_%ip" % int(m * 100)
            )
            os.makedirs(self.data_path, exist_ok=True)

            weight = base_gaba_weight * m
            for netcon in self.model.syns["I"]["con"]:
                for quanta in netcon:
                    quanta.weight[0] = weight

            self.sac_net_run(n_nets=n_nets, n_trials=n_trials, rho_steps=rho_steps)

        self.data_path = base_data_path[:]

    def vc_dir_run(
        self,
        n_trials=10,
        rhos=None,
        simultaneous=True,
        isolate_agonists=True,
        save_name=None,
        quiet=False,
    ):
        """Similar to dir_run(), but running in voltage-clamp mode to record
        current at the soma. All other inputs are blocked when recording a
        particular syanptic input. Start script with vcPas=1 to block
        membrane channels prior to voltage-clamp experiments.
        """
        if not quiet:
            print("Voltage-Clamp Directional Run. Trials: %d" % n_trials)

        n_dirs = len(self.model.dirs)
        stim = {"type": "bar", "dir": 0}
        params = self.model.get_params_dict()  # for logging
        incl_plex = self.model.n_plexus_ach > 0 and "PLEX" in self.model.synprops

        if rhos is not None:
            stim["rhos"] = rhos
            params["space_rho"] = rhos.get("space", 0)
            params["time_rho"] = rhos.get("time", 0)
            self.model.space_rho = params["space_rho"]
            self.model.time_rho = params["time_rho"]

        # create voltage clamp
        self.model.soma.push()
        h("objref VC")
        VC = h.VC
        VC = h.SEClamp(0.5)
        h.pop_section()

        # hold target voltage for entire duration
        VC.dur1 = h.tstop
        VC.dur2 = 0
        VC.dur3 = 0

        rec = h.Vector()
        rec.record(VC._ref_i)

        conditions = {
            "E": {"trans": ["E", "AMPA", "PLEX"], "holding": -60},
            "ACH": {"trans": ["E", "PLEX"], "holding": -60},
            "AMPA": {"trans": ["AMPA"], "holding": -60},
            "GABA": {"trans": ["I"], "holding": 0},
        }

        rec_data = {k: [] for k in conditions.keys()}

        for cond, settings in conditions.items():
            if not quiet:
                print("Condition: %s" % cond)

            if simultaneous:
                self.model.reset_rng()

            VC.amp1 = settings["holding"]

            for t, ps in self.model.synprops.items():
                if not incl_plex and t == "PLEX":
                    continue
                weight = (
                    ps["weight"]
                    if not isolate_agonists or t in settings["trans"]
                    else 0
                )
                for netcon in self.model.syns[t]["con"]:
                    netcon.weight = weight

            for j in range(n_trials):
                if not quiet:
                    print("trial %d..." % j, end=" ", flush=True)

                for i in range(n_dirs):
                    if not quiet:
                        print("%d" % self.model.dir_labels[i], end=" ", flush=True)

                    stim["dir"] = i

                    h.init()

                    if self.model.poisson_mode:
                        self.model.bar_poissons(stim)
                    else:
                        self.model.bar_onsets(stim)

                    self.model.update_noise()

                    rec.resize(0)
                    h.run()
                    rec_data[cond].append(np.round(rec, decimals=5))
                    self.model.clear_synapses()

                if not quiet:
                    print("")  # next line

        # reset all synaptic connection weights
        for cond, settings in conditions.items():
            for t, ps in self.model.synprops.items():
                if t != "PLEX" or incl_plex:
                    for netcon in self.model.syns[t]["con"]:
                        netcon.weight = ps["weight"]

        all_data = {
            "params": params,
            # "params": json.dumps(params),
            "soma": {
                k: self.stack_trials(n_trials, n_dirs, rec_data[k])
                for k in conditions.keys()
            },
            "syn_locs": self.model.syn_locs,
        }

        if self.model.sac_net is not None:
            all_data["sac_net"] = self.model.sac_net.get_wiring_dict()

        if save_name is not None:
            pack_hdf(os.path.join(self.data_path, save_name), all_data)
        else:
            return all_data

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

    def synaptic_density(self, all_tree=False):
        first = 0 if all_tree else self.model.first_order
        dend_lens = [d.L for order in self.model.order_list[first:] for d in order]
        n = self.model.syn_locs.shape[0]
        density = n / np.sum(dend_lens)
        print("# of synapses = %i" % n)
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
    def calc_DS(dirs: np.ndarray, response: np.ndarray):
        xpts = response * np.cos(dirs)
        ypts = response * np.sin(dirs)
        xsum = np.sum(xpts)
        ysum = np.sum(ypts)
        DSi = np.sqrt(xsum**2.0 + ysum**2.0) / (np.sum(response) + 1e-6)
        theta = np.arctan2(ysum, xsum) * 180 / np.pi

        return DSi, theta

    @staticmethod
    def stack_trials(n_trials, n_dirs, data_list: List[np.ndarray]):
        """Stack a list of run recordings [ndarrays of shape (recs, samples)]
        into a single ndarray of shape (trials, directions, recs, samples).
        """
        stack: np.ndarray = np.stack(data_list, axis=0)
        return stack.reshape(n_trials, n_dirs, *stack.shape[1:])
