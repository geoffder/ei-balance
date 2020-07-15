from neuron import h, gui

# science/math libraries
import numpy as np
import scipy.stats as st  # for probabilistic distributions

# general libraries
import platform

# local imports
from modelUtils import (
    windows_gui_fix,
    build_stim,
    map_tree,
    find_origin,
    rotate,
    merge,
)
from SacNetwork import SacNetwork
import balance_configs as configs


class DSGC:
    def __init__(self, params=None):
        self.set_default_params()
        if params is not None:
            self.update_params(params)

        if self.TTX:
            self.apply_TTX()

        if self.vc_pas:
            self.apply_vc_pas()

        self.load()  # create a new DSGC from hoc spec
        self.config()  # applies parameters

        if self.sac_mode:
            self.build_sac_net()
        else:
            self.sac_net = None

    def set_default_params(self):
        # soma physical properties
        self.soma_L    = 10
        self.soma_diam = 10
        self.soma_nseg = 1
        self.soma_Ra   = 100

        # dendrite physical properties
        self.dend_nseg   = 1
        self.seg_step    = 1 / (
            self.dend_nseg * 2) if self.dend_nseg != 10 else 0.1
        self.rec_per_sec = self.dend_nseg * 2 if self.dend_nseg != 10 else 10
        self.diam_range  = {"max": .5, "decay": 1}
        self.dend_diam   = 0.5
        self.dend_L      = 1200
        self.dend_Ra     = 100

        # global active properties
        self.TTX    = False  # zero all Na conductances
        self.vc_pas = False  # eliminate active properties for clamp

        # soma active properties
        self.active_soma    = True
        self.soma_Na        = 0.15       # [S/cm2]
        self.soma_K         = 0.035      # [S/cm2]
        self.soma_Km        = 0.003      # [S/cm2]
        self.soma_gleak_hh  = 0.0001667  # [S/cm2]
        self.soma_eleak_hh  = -60.0      # [mV]
        self.soma_gleak_pas = 0.0001667  # [S/cm2]
        self.soma_eleak_pas = -60        # [mV]

        # dend compartment active properties
        self.active_dend  = True
        self.active_terms = True  # only synapse branches and primaries

        self.prime_pas       = False      # no HHst
        self.prime_Na        = 0.20       # [S/cm2]
        self.prime_K         = 0.035      # [S/cm2]
        self.prime_Km        = 0.003      # [S/cm2]
        self.prime_gleak_hh  = 0.0001667  # [S/cm2]
        self.prime_eleak_hh  = -60.0      # [mV]
        self.prime_gleak_pas = 0.0001667  # [S/cm2]
        self.prime_eleak_pas = -60        # [mV]

        self.dend_pas       = False       # no HHst
        self.dend_Na        = 0.03        # [S/cm2]
        self.dend_K         = 0.025       # [S/cm2]
        self.dend_Km        = 0.003       # [S/cm2]
        self.dend_gleak_hh  = 0.0001667   # [S/cm2]
        self.dend_eleak_hh  = -60.0       # [mV]
        self.dend_gleak_pas = 0.0001667   # [S/cm2]
        self.dend_eleak_pas = -60         # [mV]

        # membrane noise
        self.dend_nzFactor = 0.0  # default NF_HHst = 1
        self.soma_nzFactor = 0.25

        # synaptic properties
        self.term_syn_only = True
        self.first_order   = 1  # lowest order that can hold a synapse

        self.max_quanta       = 1
        self.quanta_Pr_decay  = 0.9
        self.quanta_inter     = 5  # [ms]
        self.quanta_inter_var = 3  # [ms]

        self.NMDA_exc_lock = 0  # old excLock, remember to replace

        self.synprops = {
            "E": {
                "tau1":        0.1,    # excitatory conductance rise tau [ms]
                "tau2":        4,      # excitatory conductance decay tau [ms]
                "rev":         0,      # excitatory reversal potential [mV]
                "weight":      0.001,  # weight of exc NetCons [uS] .00023
                "prob":        0.5,    # probability of release
                "null_prob":   0.5,    # probability of release
                "pref_prob":   0.5,    # probability of release
                "delay":       0,      # [ms] mean temporal offset
                "null_offset": 0,      # [um] hard space offset on null side
                "pref_offset": 0,      # [um] hard space offset on pref side
                "var":         10,     # [ms] temporal variance of onset
            },
            "I": {
                "tau1":        0.5,    # inhibitory conductance rise tau [ms]
                "tau2":        12,     # inhibitory conductance decay tau [ms]
                "rev":         -60,    # inhibitory reversal potential [mV]
                "weight":      0.003,  # weight of inhibitory NetCons [uS]
                "prob":        0.8,    # probability of release
                "null_prob":   0.8,    # probability of release
                "pref_prob":   0.05,   # probability of release
                "delay":       -5,     # [ms] mean temporal offset
                "null_offset": -5.5,   # [um] hard space offset on null
                "pref_offset": 3,      # [um] hard space offset on pref side
                "var":         10,     # [ms] temporal variance of onset
            },
            "AMPA": {
                "tau1":        0.1,
                "tau2":        4,
                "rev":         0,
                "weight":      0.0,
                "prob":        0,
                "null_prob":   0,
                "pref_prob":   0,
                "delay":       0,
                "null_offset": 0,
                "pref_offset": 0,
                "var":         7,
            },
            "NMDA": {
                "tau1":        2,
                "tau2":        7,
                "rev":         0,
                "weight":      0.0015,
                "prob":        0,
                "null_prob":   0,
                "pref_prob":   0,
                "delay":       0,     # [ms] mean temporal offset
                "null_offset": 0,
                "pref_offset": 0,
                "var":         7,
                "n":           0.25,  # SS: 0.213
                "gama":        0.08,  # SS: 0.074
                "Voff":        0,     # 1 -> voltage independant
                "Vset":        -30,   # set voltage when independant
            },
        }
        
        self.null_rho       = 0.9
        self.pref_rho       = 0.4
        self.space_rho      = 0.9  # space correlation
        self.time_rho       = 0.9  # time correlation
        self.null_time_rho  = 0.9
        self.null_space_rho = 0.9
        self.pref_time_rho  = 0.3
        self.pref_space_rho = 0.3

        self.sac_mode           = True
        self.sac_offset         = 50
        self.sac_rho            = 0.9  # correlation of E and I dendrite angles
        self.sac_angle_rho_mode = True
        self.sac_uniform_dist   = {0: False, 1: False}  # uniform or gaussian
        self.sac_shared_var     = 30
        self.sac_theta_vars     = {"E": 60, "I": 60}
        self.sac_gaba_coverage  = 0.5
        self.sac_theta_mode     = "PN"
        
        # recording stuff
        self.downsample = {"Vm": 0.5, "iCa": .1, "g": .1}
        self.manipSyns  = [167, 152, 99, 100, 14, 93, 135, 157]  # big set
        self.manipDends = [12, 165, 98, 94, 315, 0, 284, 167]    # big set
        self.record_g   = True

        self.seed             = 0  # 10000#1
        self.nz_seed          = 0  # 10000
        self.sac_initial_seed = 0

    def update_params(self, params):
        """Update self members with key-value pairs from supplied dict."""
        self.__dict__ = merge(self.__dict__, params)

    def get_params_dict(self):
        skip = {
            "soma",
            "all_dends",
            "order_list",
            "terminals",
            "non_terms",
            "syns",
            "sac_net",
        }

        params = {
            k: v for k, v in self.__dict__.items()
            if k not in skip
        }

        return params

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

    def load(self, hoc_file="RGCmodelGD.hoc"):
        h('load_file("%s")' % hoc_file)
        self.RGC = h.DSGC(0, 0)
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

    def config(self):
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


