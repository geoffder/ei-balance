from typing import Optional
from copy import deepcopy
from neuron import h

# science/math libraries
import numpy as np
import scipy.stats as st
from deconv import quanta_to_times  # for probabilistic distributions

# local imports
from modelUtils import (
    nrn_section,
    rotate,
    merge,
    cable_dist_to_soma,
)
from SacNetwork import SacNetwork
from NetQuanta import NetQuanta
from Rig import Rig
import balance_configs as configs


class Model:
    def __init__(self, params=None):
        self.set_default_params()
        if params is not None:
            self.update_params(params)

        self.set_hoc_params()
        self.np_rng = np.random.default_rng(self.seed)

        if self.TTX:
            self.apply_TTX()

        if self.vc_pas:
            self.apply_vc_pas()

        self.create_neuron()

        self.sac_mode = False  # for Rig compatibility
        self.sac_net = None
        self.hard_offsets = {
            trans: [
                self.dir_sigmoids["offset"](d, p["null_offset"], p["pref_offset"])
                for d in self.dir_labels
            ]
            for trans, p in self.synprops.items()
        }

    def set_default_params(self):
        # hoc environment parameters
        self.tstop = 250  # [ms]
        self.steps_per_ms = 10  # [10 = 10kHz]
        self.dt = 0.1  # [ms, .1 = 10kHz]
        self.v_init = -60
        self.celsius = 36.9

        # soma physical properties
        self.soma_l = 10
        self.soma_diam = 10
        self.soma_nseg = 1
        self.soma_Ra = 100

        # dendrite physical properties
        self.dend_nseg = 1
        self.seg_step = 1 / (self.dend_nseg * 2) if self.dend_nseg != 10 else 0.1
        self.rec_per_sec = self.dend_nseg * 2 if self.dend_nseg != 10 else 10
        self.dend_diam = 0.5
        self.initial_l = 10
        self.dend_l = 120
        self.term_l = 20
        self.initial_diam = 0.5
        self.dend_diam = 0.5
        self.term_diam = 0.5
        self.dend_ra = 100

        # self.diam_scaling_mode = None  # None | "order" | "cable"
        # self.diam_range = {"max": 0.5, "min": 0.5, "decay": 1, "scale": 1}

        # global active properties
        self.TTX = False  # zero all Na conductances
        self.vc_pas = False  # eliminate active properties for clamp

        # soma active properties
        self.active_soma = True
        self.soma_na = 0.15  # [S/cm2]
        self.soma_k = 0.035  # [S/cm2]
        self.soma_km = 0.003  # [S/cm2]
        self.soma_gleak_hh = 0.0001667  # [S/cm2]
        self.soma_eleak_hh = -60.0  # [mV]
        self.soma_gleak_pas = 0.0001667  # [S/cm2]
        self.soma_eleak_pas = -60  # [mV]

        # dend compartment active properties
        self.active_dend = True

        self.initial_pas = False  # no HHst
        self.initial_na = 0.20  # [S/cm2]
        self.initial_k = 0.035  # [S/cm2]
        self.initial_km = 0.003  # [S/cm2]
        self.initial_gleak_hh = 0.0001667  # [S/cm2]
        self.initial_eleak_hh = -60.0  # [mV]
        self.initial_gleak_pas = 0.0001667  # [S/cm2]
        self.initial_eleak_pas = -60  # [mV]
        self.initial_cat = 0.0
        self.initial_cal = 0.0  # FIXME: set cat and cal

        self.dend_pas = False  # no HHst
        self.dend_na = 0.03  # [S/cm2] .03
        self.dend_k = 0.025  # [S/cm2]
        self.dend_km = 0.003  # [S/cm2]
        self.dend_gleak_hh = 0.0001667  # [S/cm2]
        self.dend_eleak_hh = -60.0  # [mV]
        self.dend_gleak_pas = 0.0001667  # [S/cm2]
        self.dend_eleak_pas = -60  # [mV]
        self.dend_cat = 0.0
        self.dend_cal = 0.0  # FIXME: set cat and cal

        self.term_pas = False
        self.term_na = 0.03  # [S/cm2] .03
        self.term_k = 0.025  # [S/cm2]
        self.term_km = 0.003  # [S/cm2]
        self.term_gleak_hh = 0.0001667  # [S/cm2]
        self.term_eleak_hh = -60.0  # [mV]
        self.term_gleak_pas = 0.0001667  # [S/cm2]
        self.term_eleak_pas = -60  # [mV]
        self.term_cat = 0.0
        self.term_cal = 0.0  # FIXME: set cat and cal

        # membrane noise
        self.dend_nz_factor = 0.0  # default NF_HHst = 1
        self.soma_nz_factor = 0.25

        # synaptic properties
        self.synprops = {
            "E": {
                "tau1": 0.1,  # excitatory conductance rise tau [ms]
                "tau2": 4,  # excitatory conductance decay tau [ms]
                "rev": 0,  # excitatory reversal potential [mV]
                "weight": 0.001,  # weight of exc NetCons [uS] .00023
                "prob": 0.5,  # probability of release
                "null_prob": 0.5,  # probability of release
                "pref_prob": 0.5,  # probability of release
                "delay": 0,  # [ms] mean temporal offset
                "null_offset": 0,  # [um] hard space offset on null side
                "pref_offset": 0,  # [um] hard space offset on pref side
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
                "null_offset": -5.5,  # [um] hard space offset on null
                "pref_offset": 3,  # [um] hard space offset on pref side
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
                "Voff": 0,  # 1 -> voltage independent
                "Vset": -30,  # set voltage when independent
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

        self.stim_onset = 50

        self.dir_labels = np.array([225, 270, 315, 0, 45, 90, 135, 180])
        self.dir_rads = np.radians(self.dir_labels)
        self.dirs = [135, 90, 45, 0, 45, 90, 135, 180]
        self.dir_inds = np.array(self.dir_labels).argsort()
        self.circle = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0])

        # take direction, null, and preferred bounds and scale between
        self.dir_sigmoids = {
            # "prob": lambda d, n, p: p
            # + (n - p) * (1 - 0.98 / (1 + np.exp(d - 91) / 25)),
            "prob": lambda d, n, p: p
            + (n - p) * (1 - 1 / (1 + np.exp((d - 90) * 0.05))),
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

        self.poisson_mode = True
        self.sac_rate = np.array([1.0])
        self.glut_rate = np.array([1.0])
        self.rate_dt = self.dt

        # recording stuff
        self.downsample = {"Vm": 0.5, "iCa": 0.1, "cai": 0.1, "g": 0.1}
        self.record_tree = True

        self.seed = 0  # 10000#1
        self.nz_seed = 0  # 10000

    def update_params(self, params):
        """Update self members with key-value pairs from supplied dict."""
        self.__dict__ = merge(self.__dict__, params)

    def reset_rng(self):
        self.nz_seed = 0
        self.np_rng = np.random.default_rng(self.seed)

    def apply_TTX(self):
        self.TTX = True
        self.soma_na = 0
        self.dend_na = 0
        self.initial_na = 0

    def apply_vc_pas(self):
        self.vc_pas = True
        self.active_soma = 0
        self.active_dend = 0
        self.active_terms = 0
        self.dend_pas = 1
        self.soma_gleak_pas = 0
        self.initial_gleak_pas = 0
        self.dend_gleak_pas = 0
        self.term_gleak_pas = 0

    def get_params_dict(self):
        skip = {
            "soma",
            "initial",
            "dend",
            "term",
            "all_dends",
            "syns",
            "dir_sigmoids",
            "dir_rads",
            "dir_inds",
            "circle",
            "np_rng",
            "syn_locs",
            "sac_net",
        }

        params = {k: v for k, v in self.__dict__.items() if k not in skip}

        return params

    def set_hoc_params(self):
        """Set hoc NEURON environment model run parameters."""
        h.finitialize()
        h.tstop = self.tstop
        h.steps_per_ms = self.steps_per_ms
        h.dt = self.dt
        h.v_init = self.v_init
        h.celsius = self.celsius

    def create_neuron(self):
        # create compartments (using parameters in self.__dict__)
        self.soma = self.create_soma()
        self.initial, self.dend, self.term = self.create_dend()
        self.all_dends = [self.initial, self.dend, self.term]
        self.initial.connect(self.soma)
        self.soma.push()
        self.create_synapse()  # generate synapses on dendrite

    def create_soma(self):
        """Build and set membrane properties of soma compartment"""
        soma = nrn_section("soma")
        soma.L = self.soma_l
        soma.diam = self.soma_diam
        soma.nseg = self.soma_nseg
        soma.Ra = self.soma_Ra

        if self.active_soma:
            soma.insert("HHst")
            soma.insert("cad")
            soma.gnabar_HHst = self.soma_na
            soma.gkbar_HHst = self.soma_k
            soma.gkmbar_HHst = self.soma_km
            soma.gleak_HHst = self.soma_gleak_hh
            soma.eleak_HHst = self.soma_eleak_hh
            soma.NF_HHst = self.soma_nz_factor
        else:
            soma.insert("pas")
            soma.g_pas = self.soma_gleak_pas
            soma.e_pas = self.soma_eleak_hh

        return soma

    def create_dend(self):
        initial = nrn_section("initial")
        dend = nrn_section("dend")
        term = nrn_section("term")

        # shared properties
        for s in [initial, dend, term]:
            s.nseg = self.dend_nseg
            s.Ra = self.dend_ra
            s.insert("HHst")
            s.insert("cad")
            s.gleak_HHst = self.dend_gleak_hh  # (S/cm2)
            s.eleak_HHst = self.dend_eleak_hh
            s.NF_HHst = self.dend_nz_factor
            s.seed_HHst = self.nz_seed

        initial.diam = self.initial_diam
        initial.L = self.initial_l
        dend.diam = self.dend_diam
        dend.L = self.dend_l
        term.diam = self.term_diam
        term.L = self.term_l

        # turn off calcium and sodium everywhere but the terminal
        for s in [initial, dend]:
            s.gtbar_HHst = 0
            s.glbar_HHst = 0
            s.gnabar_HHst = 0

        # non terminal potassium densities
        initial.gkbar_HHst = self.initial_k
        initial.gkmbar_HHst = self.initial_km
        dend.gkbar_HHst = self.dend_k
        dend.gkmbar_HHst = self.dend_km
        term.gkbar_HHst = self.term_k
        term.gkmbar_HHst = self.term_km

        # terminal active properties
        term.gnabar_HHst = self.term_na
        term.gtbar_HHst = self.term_cat
        term.glbar_HHst = self.term_cal

        dend.connect(initial)
        term.connect(dend)

        return initial, dend, term

    def create_synapse(self):
        self.term.push()
        self.syns = {}
        for trans, props in self.synprops.items():
            self.syns[trans] = {}
            if trans == "NMDA":
                self.syns[trans]["syn"] = [h.Exp2NMDA(0.5)]
                # NMDA voltage settings
                self.syns[trans]["syn"][0].n = props["n"]
                self.syns[trans]["syn"][0].gama = props["gama"]
                self.syns[trans]["syn"][0].Voff = props["Voff"]
                self.syns[trans]["syn"][0].Vset = props["Vset"]
            else:
                self.syns[trans]["syn"] = [h.Exp2Syn(0.5)]

            self.syns[trans]["syn"][0].tau1 = props["tau1"]
            self.syns[trans]["syn"][0].tau2 = props["tau2"]
            self.syns[trans]["syn"][0].e = props["rev"]

            self.syns[trans]["con"] = [
                NetQuanta(
                    self.syns[trans]["syn"][0],
                    props["weight"],
                    delay=props["delay"],
                )
            ]

        x = (self.soma_l / 2.0) + self.initial_l + self.dend_l + (self.term_l / 2.0)
        self.syn_locs = np.array([[x, 0.0, 0.0]])
        self.n_syn = 1
        h.pop_section()

    def init_synapses(self):
        """Initialize the events in each NetQuanta (NetCon wrapper)."""
        for syns in self.syns.values():
            syns["con"][0].initialize()

    def clear_synapses(self):
        for syns in self.syns.values():
            syns["con"][0].clear_events()

    def bar_poissons(self, stim):
        syn_rho = stim.get("rhos", {"time": self.time_rho})["time"]
        probs, onsets = {}, {}
        for t, props in self.synprops.items():
            if stim["type"] == "flash":
                probs[t] = props["prob"]
                onsets[t] = self.stim_onset
            else:
                # calculate probability of release
                probs[t] = self.dir_sigmoids["prob"](
                    self.dirs[stim["dir"]], props["null_prob"], props["pref_prob"]
                )
                onsets[t] = self.stim_onset + self.hard_offsets[t][stim["dir"]]

        poissons = {}
        base = self.np_rng.poisson(self.sac_rate * syn_rho)
        inv_rate = self.sac_rate * (1 - syn_rho)
        for t in ["E", "I"]:
            unshared = self.np_rng.poisson(inv_rate)
            poissons[t] = np.round((base + unshared) * probs[t]).astype(np.int)

        for t in ["AMPA", "NMDA"]:
            poissons[t] = self.np_rng.poisson(self.glut_rate * probs[t])

        for t in self.synprops.keys():
            qs = quanta_to_times(poissons[t], self.rate_dt) + onsets[t]
            for q in qs:
                self.syns[t]["con"][0].add_event(q)

    def bar_onsets(self, stim):
        for t, props in self.synprops.items():
            if stim["type"] == "flash":
                self.syns[t]["con"][0].weight = props["weight"]
                self.syns[t]["con"][0].add_event(self.stim_onset)
            else:
                # calculate probability of release
                pr = self.dir_sigmoids["prob"](
                    self.dirs[stim["dir"]], props["null_prob"], props["pref_prob"]
                )
                self.syns[t]["con"][0].weight = props["weight"] * pr
                # TODO: use dir onsets or no?
                self.syns[t]["con"][0].add_event(self.stim_onset)
                # self.syns[t]["con"][0].add_event(
                #     self.stim_onset + self.hard_offsets[t][stim["dir"]]
                # )

    def update_noise(self):
        # set HHst noise seeds
        if self.active_soma:
            self.soma.seed_HHst = self.nz_seed
            self.nz_seed += 1

        if self.active_dend:
            self.initial.seed_HHst = self.nz_seed
            self.nz_seed += 1
            self.dend.seed_HHst = self.nz_seed
            self.nz_seed += 1
            self.term.seed_HHst = self.nz_seed
            self.nz_seed += 1

    def get_recording_locations(self):
        locs = []
        # NOTE: hardcoded assuming 2 per section (start and 0.5) for now
        # per = self.rec_per_sec
        initial_x = self.soma_l / 2.0
        dend_x = initial_x + self.initial_l
        term_x = dend_x + self.dend_l
        locs = [
            initial_x,
            initial_x + (self.initial_l / 2),
            dend_x,
            dend_x + (self.dend_l / 2),
            term_x,
            term_x + (self.term_l / 2),
        ]

        return np.array(locs)
