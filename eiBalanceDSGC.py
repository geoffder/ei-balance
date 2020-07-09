from neuron import h, gui

# science/math libraries
import numpy as np
import scipy.stats as st  # for probabilistic distributions

# general libraries
import platform

# local imports
from modelUtils import (
    windows_gui_fix,
    nrn_objref,
    build_stim,
    map_tree,
    find_origin,
    rotate,
    merge,
)
from SacNetwork import SacNetwork
from experiments import Rig
import balance_configs as configs


class Model:
    def __init__(self, params=None):
        self.set_default_params()
        if params is not None:
            self.update_params(params)
        self.set_hoc_params()

        if self.TTX:
            self.apply_TTX()

        if self.vc_pas:
            self.apply_vc_pas()

        self.load_DSGC()  # create a new DSGC from hoc spec
        self.config_DSGC()  # applies parameters

        if self.sac_mode:
            self.build_sac_net()
        else:
            self.sac_net = None

    def set_default_params(self):
        # hoc environment parameters
        self.tstop = 250  # [ms]
        self.steps_per_ms = 10  # [10 = 10kHz]
        self.dt = 0.1  # [ms, .1 = 10kHz]
        self.v_init = -60
        self.celsius = 36.9

        # soma physical properties
        self.soma_L = 10
        self.soma_diam = 10
        self.soma_nseg = 1
        self.soma_Ra = 100

        # dendrite physical properties
        self.dend_nseg = 1
        self.seg_step = 1 / (
            self.dend_nseg * 2) if self.dend_nseg != 10 else 0.1
        self.rec_per_sec = self.dend_nseg * 2 if self.dend_nseg != 10 else 10
        self.diam_range = {"max": .5, "decay": 1}
        self.dend_diam = 0.5
        self.dend_L = 1200
        self.dend_Ra = 100

        # global active properties
        self.TTX = False  # zero all Na conductances
        self.vc_pas = False  # eliminate active properties for clamp

        # soma active properties
        self.active_soma = True
        self.soma_Na = 0.15  # [S/cm2]
        self.soma_K = 0.035  # [S/cm2]
        self.soma_Km = 0.003  # [S/cm2]
        self.soma_gleak_hh = 0.0001667  # [S/cm2]
        self.soma_eleak_hh = -60.0  # [mV]
        self.soma_gleak_pas = 0.0001667  # [S/cm2]
        self.soma_eleak_pas = -60  # [mV]

        # dend compartment active properties
        self.active_dend = True
        self.active_terms = True  # only synapse branches and primaries

        self.prime_pas = False  # no HHst
        self.prime_Na = 0.20  # [S/cm2]
        self.prime_K = 0.035  # [S/cm2]
        self.prime_Km = 0.003  # [S/cm2]
        self.prime_gleak_hh = 0.0001667  # [S/cm2]
        self.prime_eleak_hh = -60.0  # [mV]
        self.prime_gleak_pas = 0.0001667  # [S/cm2]
        self.prime_eleak_pas = -60  # [mV]

        self.dend_pas = False  # no HHst
        self.dend_Na = 0.03  # [S/cm2] .03
        self.dend_K = 0.025  # [S/cm2]
        self.dend_Km = 0.003  # [S/cm2]
        self.dend_gleak_hh = 0.0001667  # [S/cm2]
        self.dend_eleak_hh = -60.0  # [mV]
        self.dend_gleak_pas = 0.0001667  # [S/cm2]
        self.dend_eleak_pas = -60  # [mV]

        # membrane noise
        self.dend_nzFactor = 0.0  # default NF_HHst = 1
        self.soma_nzFactor = 0.25

        # synaptic properties
        self.term_syn_only = True
        self.first_order = 1  # lowest order that can hold a synapse

        self.flash_mean = 100  # mean onset time for flash stimulus
        self.flash_variance = 400  # variance of flash release onset
        self.jitter = 60  # [ms] 10

        self.max_quanta = 1
        self.quanta_Pr_decay = 0.9
        self.quanta_inter = 5  # [ms]
        self.quanta_inter_var = 3

        self.NMDA_exc_lock = 0  # old excLock, remember to replace

        self.synprops = {
            "E": {
                "tau1": 0.1,  # excitatory conductance rise tau [ms]
                "tau2": 4,  # excitatory conductance decay tau [ms]
                "rev": 0,  # excitatory reversal potential [mV]
                "weight": 0.001,  # weight of excitatory NetCons [uS] .00023
                "prob": 0.5,  # probability of release
                "null_prob": 0.5,  # probability of release
                "pref_prob": 0.5,  # probability of release
                "delay": 0,  # [ms] mean temporal offset
                "null_offset": 0,  # [um] hard-coded space offset on null side
                "pref_offset": 0,  # [um] hard-coded space offset on pref side
                "var": 10,  # [ms] temporal variance of onset
            },
            "I": {
                "tau1": 0.5,  # inhibitory conductance rise tau [ms]
                "tau2": 12,  # inhibitory conductance decay tau [ms]
                "rev": -60,  # inhibitory reversal potential [mV]
                "weight": 0.003,  # weight of inhibitory NetCons [uS]
                "prob": 0.8,  # probability of release
                "null_prob": 0.8,  # probability of release
                "pref_prob": 0.05,  # probability of release
                "delay": -5,  # [ms] mean temporal offset
                "null_offset": -5.5,  # [um] hard-coded space offset on null
                "pref_offset": 3,  # [um] hard-coded space offset on pref side
                "var": 10,  # [ms] temporal variance of onset
            },
            "AMPA": {
                "tau1": 0.1,
                "tau2": 4,
                "rev": 0,
                "weight": 0.0,
                "prob": 0,
                "null_prob": 0,
                "pref_prob": 0,
                "delay": 0,
                "null_offset": 0,
                "pref_offset": 0,
                "var": 7,
            },
            "NMDA": {
                "tau1": 2,
                "tau2": 7,
                "rev": 0,
                "weight": 0.0015,
                "prob": 0,
                "null_prob": 0,
                "pref_prob": 0,
                "delay": 0,  # [ms] mean temporal offset
                "null_offset": 0,
                "pref_offset": 0,
                "var": 7,
                "n": 0.25,  # SS: 0.213
                "gama": 0.08,  # SS: 0.074
                "Voff": 0,  # 1 -> voltage independant
                "Vset": -30,  # set voltage when independant
            },
        }

        # light stimulus
        self.light_bar = {
            "start_time": 0,
            "speed": 1.0,  # 1#1.6 # speed of the stimulus bar (um/ms)
            "width": 250,  # width of the stimulus bar(um)
            "x_motion": True,  # move bar in x, if not, move bar in y
            "x_start": -40,  # start location (X axis) of the stim bar (um)
            "x_end": 200,  # end location (X axis)of the stimulus bar (um)
            "y_start": 25,  # start location (Y axis) of the stimulus bar (um)
            "y_end": 225,  # end location (Y axis) of the stimulus bar (um)
        }

        self.dir_labels = [225, 270, 315, 0, 45, 90, 135, 180]
        self.dir_rads = np.radians(self.dir_labels)
        self.dirs = [135, 90, 45, 0, 45, 90, 135, 180]
        self.dir_inds = np.array(self.dir_labels).argsort()
        self.circle = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0])

        # take direction, null, and preferred bounds and scale between
        self.dir_sigmoids = {
            "prob": lambda d, n, p: p
            + (n - p) * (1 - 0.98 / (1 + np.exp(d - 91) / 25)),
            "offset": lambda d, n, p: p
            + (n - p) * (1 - 0.98 / (1 + np.exp(d - 74.69) / 24.36)),
        }
        self.null_rho = 0.9
        self.pref_rho = 0.4
        self.space_rho = 0.9  # space correlation
        self.time_rho = 0.9  # time correlation
        self.null_time_rho = 0.9
        self.null_space_rho = 0.9
        self.pref_time_rho = 0.3
        self.pref_space_rho = 0.3

        self.sac_mode = True
        self.sac_offset = 50
        self.sac_rho = 0.9  # correlation of E and I dendrite angles
        self.sac_angle_rho_mode = True
        self.sac_uniform_dist = {0: False, 1: False}  # uniform or gaussian
        self.sac_shared_var = 30
        self.sac_theta_vars = {"E": 60, "I": 60}
        self.sac_gaba_coverage = 0.5
        self.sac_theta_mode = "PN"
        
        # recording stuff
        self.downsample = {"Vm": 0.5, "iCa": .1, "g": .1}
        self.manipSyns = [167, 152, 99, 100, 14, 93, 135, 157]  # big set
        self.manipDends = [12, 165, 98, 94, 315, 0, 284, 167]  # big set
        self.record_g = True

        self.seed = 0  # 10000#1
        self.nz_seed = 0  # 10000
        self.sac_initial_seed = 0

    def update_params(self, params):
        """Update self members with key-value pairs from supplied dict."""
        self.__dict__ = merge(self.__dict__, params)

    def apply_TTX(self):
        self.TTX = True
        self.soma_Na = 0
        self.dend_Na = 0
        self.prime_Na = 0

    def apply_vc_pas(self):
        self.vc_pas = True
        self.active_soma = 0
        self.active_dend = 0
        self.active_terms = 0
        self.dend_pas = 1
        self.soma_gleak_pas = 0
        self.dend_gleak_pas = 0
        self.prime_gleak_pas = 0

    def get_params_dict(self):
        skip = {
            "RGC",
            "soma",
            "all_dends",
            "order_list",
            "terminals",
            "non_terms",
            "syns",
            "sac_net",
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

    def set_hoc_params(self):
        """Set hoc NEURON environment model run parameters."""
        h.tstop = self.tstop
        h.steps_per_ms = self.steps_per_ms
        h.dt = self.dt
        h.v_init = self.v_init
        h.celsius = self.celsius

    def load_DSGC(self):
        self.RGC = nrn_objref("RGC")
        self.RGC = h.DSGC(0, 0)  # h("RGC = new DSGC(0,0)")
        self.soma = self.RGC.soma
        self.all_dends = self.RGC.dend

        self.origin = find_origin(self.all_dends)
        self.soma.push()

        self.order_list, self.terminals, self.non_terms = map_tree(self.RGC)
        print("Number of terminal dendrites: %d" % len(self.terminals))

    def config_soma(self):
        """Build and set membrane properties of soma compartment"""
        self.soma.L = self.soma_L
        self.soma.diam = self.soma_diam
        self.soma.nseg = self.soma_nseg
        self.soma.Ra = self.soma_Ra

        if self.active_soma:
            self.soma.insert("HHst")
            self.soma.gnabar_HHst = self.soma_Na
            self.soma.gkbar_HHst = self.soma_K
            self.soma.gkmbar_HHst = self.soma_Km
            self.soma.gleak_HHst = self.soma_gleak_hh
            self.soma.eleak_HHst = self.soma_eleak_hh
            self.soma.NF_HHst = self.soma_nzFactor
        else:
            self.soma.insert("pas")
            self.soma.g_pas = self.soma_gleak_pas
            self.soma.e_pas = self.soma_eleak_hh

    def config_dends(self):
        """Set membrane properties of dendrite compartments"""
        if self.active_terms:
            for dend in self.terminals:
                dend.insert("HHst")
                dend.gnabar_HHst = self.dend_Na
                dend.gkbar_HHst = self.dend_K
                dend.gkmbar_HHst = self.dend_Km
                dend.gleak_HHst = self.dend_gleak_hh
                dend.eleak_HHst = self.dend_eleak_hh
                dend.NF_HHst = self.dend_nzFactor
            if self.dend_pas:
                for dend in self.non_terms:
                    dend.insert("pas")
                    dend.g_pas = self.dend_gleak_pas
                    dend.e_pas = self.dend_eleak_pas
            else:
                for dend in self.non_terms:
                    dend.insert("HHst")
                    dend.gnabar_HHst = 0
                    dend.gkbar_HHst = self.dend_K
                    dend.gkmbar_HHst = self.dend_Km
                    dend.gleak_HHst = self.dend_gleak_hh
                    dend.eleak_HHst = self.dend_eleak_hh
                    dend.NF_HHst = self.dend_nzFactor
        else:
            for order in self.order_list[1:]:  # except primes
                if self.active_dend:
                    for dend in order:
                        dend.insert("HHst")
                        dend.gnabar_HHst = self.dend_Na
                        dend.gkbar_HHst = self.dend_K
                        dend.gkmbar_HHst = self.dend_Km
                        dend.gleak_HHst = self.dend_gleak_hh
                        dend.eleak_HHst = self.dend_eleak_hh
                elif self.dend_pas:
                    for dend in order:
                        dend.insert("pas")
                        dend.g_pas = self.dend_gleak_pas
                        dend.e_pas = self.dend_eleak_pas
                else:
                    for dend in order:
                        dend.insert("HHst")
                        dend.gnabar_HHst = 0
                        dend.gkbar_HHst = self.dend_K
                        dend.gkmbar_HHst = self.dend_Km
                        dend.gleak_HHst = self.dend_gleak_hh
                        dend.eleak_HHst = self.dend_eleak_hh

        # prime dendrites
        for dend in self.order_list[0]:
            if self.active_dend:
                dend.insert("HHst")
                dend.gnabar_HHst = self.prime_Na
                dend.gkbar_HHst = self.prime_K
                dend.gkmbar_HHst = self.prime_Km
                dend.gleak_HHst = self.prime_gleak_hh
                dend.eleak_HHst = self.prime_eleak_hh
            else:
                dend.insert("pas")
                dend.g_pas = self.prime_gleak_pas
                dend.e_pas = self.prime_eleak_pas

        # all dendrites
        for dend in self.all_dends:
            dend.nseg = self.dend_nseg
            dend.Ra = 100
            if self.active_dend:
                dend.gtbar_HHst = 0.0003  # default
                dend.glbar_HHst = 0.0003  # default
                dend.NF_HHst = self.dend_nzFactor

        # set dendrite diameters
        diam = self.diam_range["max"]
        for i, order in enumerate(self.order_list):
            for dend in order:
                dend.diam = diam
            diam *= self.diam_range["decay"]
            
    def create_synapses(self):
        # number of synapses
        h("n_syn = 0")
        if self.term_syn_only:
            h.n_syn = len(self.terminals)
        else:
            for order in self.order_list[self.first_order:]:
                h.n_syn += len(order)
        self.n_syn = int(h.n_syn)

        # create *named* hoc objects for each synapse (for gui compatibility)
        h("objref e_syns[n_syn], i_syns[n_syn]")
        h("objref nmda_syns[n_syn], ampa_syns[n_syn]")

        # complete synapses are made up of a NetStim, Syn, and NetCon
        self.syns = {
            "X": [],
            "Y": [],
            "E": {"stim": [], "syn": h.e_syns, "con": []},
            "I": {"stim": [], "syn": h.i_syns, "con": []},
            "NMDA": {"stim": [], "syn": h.nmda_syns, "con": []},
            "AMPA": {"stim": [], "syn": h.ampa_syns, "con": []},
        }

        if self.term_syn_only:
            dend_list = self.terminals
        else:
            dend_list = [
                dend
                for order in self.order_list[self.first_order:]
                for dend in order
            ]

        for i, dend in enumerate(dend_list):
            dend.push()

            # 3D location of the synapses on this dendrite
            # place them in the middle since only one syn per dend
            pts = int(h.n3d())
            if pts % 2:  # odd number of points
                self.syns["X"].append(h.x3d((pts - 1) / 2))
                self.syns["Y"].append(h.y3d((pts - 1) / 2))
            else:
                self.syns["X"].append(
                    (h.x3d(pts / 2) + h.x3d((pts / 2) - 1)) / 2.0)
                self.syns["Y"].append(
                    (h.y3d(pts / 2) + h.y3d((pts / 2) - 1)) / 2.0)

            for trans, props in self.synprops.items():
                if trans == "NMDA":
                    self.syns[trans]["syn"][i] = h.Exp2NMDA(0.5)
                    # NMDA voltage settings
                    self.syns[trans]["syn"][i].n = props["n"]
                    self.syns[trans]["syn"][i].gama = props["gama"]
                    self.syns[trans]["syn"][i].Voff = props["Voff"]
                    self.syns[trans]["syn"][i].Vset = props["Vset"]
                else:
                    self.syns[trans]["syn"][i] = h.Exp2Syn(0.5)

                self.syns[trans]["syn"][i].tau1 = props["tau1"]
                self.syns[trans]["syn"][i].tau2 = props["tau2"]
                self.syns[trans]["syn"][i].e = props["rev"]

                # create NetStims to drive the synapses through NetCons
                self.syns[trans]["stim"].append(
                    [build_stim() for _ in range(self.max_quanta)]
                )

                # NOTE: Legacy used 10ms delay here (second int param)
                self.syns[trans]["con"].append(
                    [
                        h.NetCon(
                            stim, self.syns[trans]["syn"][i], 0, 0,
                            props["weight"]
                        )
                        for stim in self.syns[trans]["stim"][i]
                    ]
                )

            h.pop_section()

    def config_DSGC(self):
        self.config_soma()
        self.config_dends()
        self.create_synapses()

        # pre-calculations based on set properties
        self.hard_offsets = {
            trans: [
                self.dir_sigmoids["offset"](
                    d, p["pref_offset"], p["null_offset"]
                )
                for d in self.dir_labels
            ]
            for trans, p in self.synprops.items()
        }

    def build_sac_net(self, rho=None, initial_seed=None):
        if rho is not None:
            self.sac_rho = rho

        if initial_seed is not None:
            self.sac_initial_seed = initial_seed

        probs = {
            t: {
                "null": self.synprops[t]["null_prob"],
                "pref": self.synprops[t]["pref_prob"],
            }
            for t in ["E", "I"]
        }

        self.sac_net = SacNetwork(
            {"X": self.syns["X"], "Y": self.syns["Y"]},
            probs,
            self.sac_rho,
            self.sac_uniform_dist,
            self.sac_shared_var,
            self.sac_theta_vars,
            self.sac_gaba_coverage,
            self.dir_labels,
            self.sac_offset,
            self.sac_initial_seed,
            self.sac_theta_mode,
        )

    def flash_onsets(self):
        """
        Calculate onset times for each synapse randomly as though stimulus was
        a flash. Timing jitter is applied with pseudo-random number generators.
        """
        # distribution to pull mean onset times for each site from
        mOn = h.Random(self.seed)
        mOn.normal(self.flash_mean, self.flash_variance)
        self.seed += 1

        # store timings for return (fine to ignore return)
        onset_times = {trans: [] for trans in ["E", "I", "AMPA", "NMDA"]}

        rand_on = {trans: 0 for trans in ["E", "I", "AMPA", "NMDA"]}

        for s in range(self.n_syn):
            onset = mOn.repick()

            for t in self.synprops.keys():
                r = h.Random(self.seed)
                r.normal(0, 1)
                self.seed += 1
                rand_on[t] = r.repick()

            rand_on["E"] = rand_on["I"] * self.time_rho + (
                rand_on["E"] * np.sqrt(1 - self.time_rho ** 2)
            )

            for q in range(self.max_quanta):
                if q:
                    # add variable delay til next quanta (if there is one)
                    quanta_delay = h.Random(self.seed)
                    quanta_delay.normal(
                        self.quanta_inter, self.quanta_inter_var
                    )
                    self.seed += 1
                    onset += quanta_delay.repick()

                for t, props in self.synprops.items():
                    self.syns[t]["stim"][s][q].start = (
                        onset + props["delay"] + rand_on[t] * props["var"]
                    )

                    onset_times[t].append(rand_on[t])

                if self.NMDA_exc_lock:
                    self.syns["NMDA"]["stim"][s][q].start = (
                        self.syns["E"]["stim"][s][q].start
                    )

        return onset_times

    def bar_sweep(self, syn, dir_idx):
        """Return activation time for the given synapse based on the light bar
        config and the pre-synaptic setup (e.g. SAC network/hard-coded offsets)
        """
        bar = self.light_bar
        rotation = -self.dir_rads[dir_idx]
        x, y = rotate(
            self.origin, self.syns["X"][syn], self.syns["Y"][syn], rotation
        )
        loc = {"x": x, "y": y}
        ax = "x" if bar["x_motion"] else "y"

        on_times = {}

        # distance to synapse divided by speed
        for t, props in self.synprops.items():
            if t in ["E", "I"] and self.sac_net is not None:
                sac_loc = self.sac_net.get_syn_loc(t, syn, rotation)
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
        """
        sac = self.sac_net

        time_rho = stim.get("rhos", {"time": self.time_rho})["time"]

        # store timings for return (fine to ignore return)
        onset_times = {trans: [] for trans in ["E", "I", "AMPA", "NMDA"]}

        rand_on = {trans: 0 for trans in ["E", "I", "AMPA", "NMDA"]}

        for s in range(self.n_syn):
            bar_times = self.bar_sweep(s, stim["dir"])
            if (sac is None
                    or not self.sac_angle_rho_mode
                    or not sac.gaba_here[s]):
                syn_rho = time_rho
            else:
                # scale correlation of E and I by diff of their dend angles
                delta = (sac.thetas["E"][s] - sac.thetas["I"][s]) % 360
                delta = 360 - delta if delta > 180 else delta
                syn_rho = time_rho - time_rho * delta / 180

            # shared jitter
            jit_rand = h.Random(self.seed)
            jit_rand.normal(0, 1)
            self.seed += 1
            jit = jit_rand.repick()
            bar_times = {
                k: v + jit * self.jitter
                for k, v in bar_times.items()
            }

            for t in self.synprops.keys():
                rand_on[t] = h.Random(self.seed).normal(0, 1)
                self.seed += 1

            rand_on["E"] = rand_on["I"] * syn_rho + (
                rand_on["E"] * np.sqrt(1 - syn_rho ** 2)
            )

            for q in range(self.max_quanta):
                if q:
                    # add variable delay til next quanta (if there is one)
                    quanta_rand = h.Random(self.seed)
                    quanta_rand.normal(
                        self.quanta_inter, self.quanta_inter_var
                    )
                    self.seed += 1
                    q_delay = quanta_rand.repick()
                    bar_times = {k: v + q_delay for k, v in bar_times.items()}

                for t, props in self.synprops.items():
                    self.syns[t]["stim"][s][q].start = (
                        bar_times[t] + props["delay"]
                        + rand_on[t] * props["var"]
                    )

                    onset_times[t].append(rand_on[t])

                # TODO: Maybe remove this option since I don't think it's used
                if self.NMDA_exc_lock:
                    self.syns["NMDA"]["stim"][s][q].start = (
                        self.syns["E"]["stim"][s][q].start
                    )

        if locked_synapses is not None:
            # reset activation time to coincide with bar (no noise)
            for s in locked_synapses["nums"]:
                lock_times = self.bar_sweep(s, stim["dir"])
                for t in ["E", "I"]:
                    self.model.syns[t]["stim"][s][0].start = lock_times[t]

        return onset_times

    def set_failures(self, stim, locked_synapses=None):
        """
        Determine number of quantal activations of each synapse occur on a
        trial. Psuedo-random numbers generated for each synapse are compared
        against thresholds set by probability of release to determine if the
        "pre-synapse" succeeds or fails to release neurotransmitter.
        """
        sac = self.sac_net

        space_rho = stim.get("rhos", {"space": self.space_rho})["space"]

        # numbers above can result in NaNs
        rho = 0.986 if space_rho > 0.986 else space_rho

        # calculate input rho required to achieve the desired output rho
        # exponential fit: y = y0 + A * exp(-invTau * x)
        # y0 = 1.0461; A = -0.93514; invTau = 3.0506
        rho = 1.0461 - 0.93514 * np.exp(-3.0506 * rho)

        rand_succ = {}
        for t in self.synprops.keys():
            rand_succ[t] = h.Random(self.seed)
            rand_succ[t].normal(0, 1)
            self.seed += 1

        picks = {
            t: np.array([rand_succ[t].repick() for i in range(self.n_syn)])
            for t in self.synprops.keys()
        }

        # correlate synaptic variance of ACH with GABA
        if sac is None or not self.sac_angle_rho_mode:
            picks["E"] = picks["I"] * rho + (
                picks["E"] * np.sqrt(1 - rho ** 2))
        else:
            # scale correlation of E and I by prox of their dend angles
            for i in range(self.n_syn):
                if sac.gaba_here[i]:
                    delta = abs(sac.thetas["E"][i] - sac.thetas["I"][i]) % 360
                    delta = 360 - delta if delta > 180 else delta
                    syn_rho = rho - rho * delta / 180
                    picks["E"][i] = (
                        picks["I"][i] * syn_rho
                        + picks["E"][i] * np.sqrt(1 - syn_rho ** 2)
                    )

        probs = {}
        for t, props in self.synprops.items():
            if stim["type"] == "flash":
                probs[t] = np.full(self.n_syn, props["prob"])
            elif sac is not None and t in ["E", "I"]:
                probs[t] = sac.probs[t][:, stim["dir"]]
            else:
                # calculate probability of release
                p = self.dir_sigmoids["prob"](
                    self.dirs[stim["dir"]],
                    props["pref_prob"],
                    props["null_prob"]
                )
                probs[t] = np.full(self.n_syn, p)

        sdevs = {k: np.std(v) for k, v in picks.items()}
        successes = {}

        for t in picks.keys():
            # determine release bool for each possible quanta
            q_probs = np.array([
                probs[t] * (self.quanta_Pr_decay ** q)
                for q in range(self.max_quanta)
            ])
            left = st.norm.ppf((1 - q_probs) / 2.0) * sdevs[t]
            right = st.norm.ppf(1 - (1 - q_probs) / 2.0) * sdevs[t]
            successes[t] = (left < picks[t]) * (picks[t] < right)

            for i in range(self.n_syn):
                for q in range(self.max_quanta):
                    if successes[t][q][i]:
                        self.syns[t]["stim"][i][q].number = 1
                    else:
                        self.syns[t]["stim"][i][q].number = 0

        if locked_synapses is not None:
            for s in locked_synapses["nums"]:
                self.syns[t]["con"][s][0].weight[0] = (
                    self.synprops["I"]["weight"] * 10)
                for t in ["E", "I"]:
                    self.syns[t]["stim"][s][0].number = locked_synapses[t]

        pearson = st.pearsonr(
            successes["E"].flatten(), successes["I"].flatten()
        )
        return pearson[0]

    def update_noise(self):
        # set HHst noise seeds
        if self.active_soma:
            self.soma.seed_HHst = self.nz_seed
            self.nz_seed += 1

        if self.active_terms:
            for dend in self.terminals:
                dend.seed_HHst = self.nz_seed
                self.nz_seed += 1

        elif self.active_dend:
            for order in self.order_list[1:]:  # except primes
                for dend in order:
                    dend.seed_HHst = self.nz_seed
                    self.nz_seed += 1

        # prime dendrites
        if self.active_dend:  # regardless if active_terms
            for dend in self.order_list[0]:
                dend.seed_HHst = self.nz_seed
                self.nz_seed += 1

    def get_recording_locations(self):
        locs = {"X": [], "Y": []}
        per = self.rec_per_sec
        for dend in self.all_dends:
            dend.push()
            pts = int(h.n3d())

            # Rough coordinates based on indexing the list of 3d points
            # assigned to the current section. Number of points vary.
            for s in range(per):
                locs["X"].append(h.x3d(s * (pts - 1) / per))
                locs["Y"].append(h.y3d(s * (pts - 1) / per))

            h.pop_section()

        return np.array([locs["X"], locs["Y"]])


if __name__ == "__main__":
    plat = platform.system()
    if plat == "Linux":
        h('load_file("RGCmodelGD.hoc")')
        basest = "/mnt/Data/NEURONoutput/"
    else:
        h('load_file("RGCmodelGD.hoc")')
        basest = "D:\\NEURONoutput\\"
        windows_gui_fix()

    h.xopen("new_balance.ses")  # open neuron gui session

    dsgc = Model(configs.sac_mode_config())
    # dsgc = Model(configs.offset_mode_config())
    rig = Rig(dsgc, basest)
    rig.synaptic_density()
    
    # rig.dir_run(3)
    rig.sac_net_run(n_nets=5, n_trials=3, rho_steps=[0., 1.])
    # rig.vc_dir_run(10)
    # dsgc.sac_net.plot_dends(0)
    # locs = dsgc.get_recording_locations()
    # rig.offset_run(1)

    # dist = rig.sac_angle_distribution()
    # dist.savefig(basest + "angle_dist.png", bbox_inches="tight")
