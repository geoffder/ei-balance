from typing import Optional
from copy import deepcopy
from neuron import h

# science/math libraries
import numpy as np
import scipy.stats as st
from deconv import (
    poisson_of_release,
    quanta_to_times,
)  # for probabilistic distributions

# local imports
from modelUtils import (
    map_tree,
    find_origin,
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

        self.load_DSGC()  # create a new DSGC from hoc spec
        self.config_DSGC()  # applies parameters

        if self.sac_mode:
            self.build_sac_net()
            if self.n_plexus_ach > 0 and "PLEX" not in self.synprops:
                self.synprops["PLEX"] = deepcopy(self.synprops["E"])
            elif self.n_plexus_ach <= 0:
                del self.synprops["PLEX"]
        else:
            self.sac_net: Optional[SacNetwork] = None
            if "PLEX" in self.synprops:
                del self.synprops["PLEX"]

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
        self.seg_step = 1 / (self.dend_nseg * 2) if self.dend_nseg != 10 else 0.1
        self.rec_per_sec = self.dend_nseg * 2 if self.dend_nseg != 10 else 10
        self.dend_diam = 0.5
        self.dend_L = 1200
        self.dend_Ra = 100

        self.diam_scaling_mode = None  # None | "order" | "cable"
        self.diam_range = {"max": 0.5, "min": 0.5, "decay": 1, "scale": 1}

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

        self.max_quanta = 1
        self.quanta_Pr_decay = 0.9
        self.quanta_inter = 5  # [ms]
        self.quanta_inter_var = 3  # [ms]

        self.NMDA_exc_lock = 0  # old excLock, remember to replace

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
            "PLEX": {
                "tau1": 0.1,  # excitatory conductance rise tau [ms]
                "tau2": 4,  # excitatory conductance decay tau [ms]
                "rev": 0,  # excitatory reversal potential [mV]
                "weight": 0.00025,  # weight of exc NetCons [uS] .00023
                "prob": 0.5,  # probability of release
                "null_prob": 0.5,  # probability of release
                "pref_prob": 0.5,  # probability of release
                "delay": 0,  # [ms] mean temporal offset
                "null_offset": 0,  # [um] hard space offset on null side
                "pref_offset": 0,  # [um] hard space offset on pref side
                "var": 10,  # [ms] temporal variance of onset
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

        self.flash_mean = 100  # mean onset time for flash stimulus
        self.flash_variance = 400  # variance of flash release onset
        self.jitter = 60  # [ms] 10

        self.dir_labels = np.array([225, 270, 315, 0, 45, 90, 135, 180])
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
        self.n_plexus_ach = 0

        self.sac_rate = np.array([1.0])
        self.glut_rate = np.array([1.0])

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
            "np_rng",
            "syn_locs",
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

    def load_DSGC(self):
        _ = h.load_file("RGCmodelGD.hoc")
        self.RGC = h.DSGC(0, 0)
        self.soma = self.RGC.soma
        self.all_dends = self.RGC.dend
        self.origin = find_origin(self.all_dends)
        self.soma.push()
        self.order_list, self.terminals, self.non_terms = map_tree(self.RGC)

    def config_soma(self):
        """Build and set membrane properties of soma compartment"""
        self.soma.L = self.soma_L
        self.soma.diam = self.soma_diam
        self.soma.nseg = self.soma_nseg
        self.soma.Ra = self.soma_Ra

        if self.active_soma:
            self.soma.insert("HHst")
            self.soma.insert("cad")
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
                dend.insert("cad")
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
                    dend.insert("cad")
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
                        dend.insert("cad")
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
                        dend.insert("cad")
                        dend.gnabar_HHst = 0
                        dend.gkbar_HHst = self.dend_K
                        dend.gkmbar_HHst = self.dend_Km
                        dend.gleak_HHst = self.dend_gleak_hh
                        dend.eleak_HHst = self.dend_eleak_hh

        # prime dendrites
        for dend in self.order_list[0]:
            if self.active_dend:
                dend.insert("HHst")
                dend.insert("cad")
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
        if self.diam_scaling_mode == "order":
            diam = self.diam_range["max"]
            for order in self.order_list:
                for dend in order:
                    dend.diam = diam
                diam *= self.diam_range["decay"]
        elif self.diam_scaling_mode == "cable":
            dists = cable_dist_to_soma(self)
            dist_max = np.max(dists)
            dist_min = np.min(dists)
            dist_range = dist_max - dist_min
            for dist, dend in zip(dists, self.all_dends):
                d = self.diam_range["max"] - (
                    self.diam_range["max"] - self.diam_range["min"]
                ) * min(self.diam_range["scale"] * (dist - dist_min) / dist_range, 1)
                # print(d)
                dend.diam = d

    def create_synapses(self):
        # number of synapses
        if self.term_syn_only:
            self.n_syn = len(self.terminals)
        else:
            self.n_syn = 0
            for order in self.order_list[self.first_order :]:
                self.n_syn += len(order)

        # complete synapses are made up of a NetStim, Syn, and NetCon
        locs = []
        self.syns = {
            "E": {"syn": [], "con": []},
            "I": {"syn": [], "con": []},
            "NMDA": {"syn": [], "con": []},
            "AMPA": {"syn": [], "con": []},
        }

        if self.term_syn_only:
            dend_list = self.terminals
        else:
            dend_list = [
                dend for order in self.order_list[self.first_order :] for dend in order
            ]

        for i, dend in enumerate(dend_list):
            dend.push()

            # 3D location of the synapses on this dendrite
            # place them in the middle since only one syn per dend
            pts = int(h.n3d())
            if pts % 2:  # odd number of points
                u = (pts - 1) / 2
                locs.append([h.x3d(u), h.y3d(u), h.z3d(u)])
            else:
                u1 = pts / 2
                u2 = (pts - 1) / 2
                locs.append(
                    [
                        (h.x3d(u1) + h.x3d(u2)) / 2.0,
                        (h.y3d(u1) + h.y3d(u2)) / 2.0,
                        (h.z3d(u1) + h.z3d(u2)) / 2.0,
                    ]
                )

            for trans, props in self.synprops.items():
                if trans == "PLEX":
                    continue
                if trans == "NMDA":
                    self.syns[trans]["syn"].append(h.Exp2NMDA(0.5))
                    # NMDA voltage settings
                    self.syns[trans]["syn"][i].n = props["n"]
                    self.syns[trans]["syn"][i].gama = props["gama"]
                    self.syns[trans]["syn"][i].Voff = props["Voff"]
                    self.syns[trans]["syn"][i].Vset = props["Vset"]
                else:
                    self.syns[trans]["syn"].append(h.Exp2Syn(0.5))

                self.syns[trans]["syn"][i].tau1 = props["tau1"]
                self.syns[trans]["syn"][i].tau2 = props["tau2"]
                self.syns[trans]["syn"][i].e = props["rev"]

                # TODO: instead of having max_quanta stim "slots", and setting the time
                # (and number 0 or 1) of events for each synapse, just add events to the NetQuanta
                # NetQuanta (wraps NetCon) for scheduling and applying conductance events
                self.syns[trans]["con"].append(
                    NetQuanta(
                        self.syns[trans]["syn"][i],
                        props["weight"],
                        delay=props["delay"],
                    )
                )
            h.pop_section()
        self.syn_locs = np.array(locs)

    def init_synapses(self):
        """Initialize the events in each NetQuanta (NetCon wrapper)."""
        for syns in self.syns.values():
            for nq in syns["con"]:
                nq.initialize()

    def clear_synapses(self):
        for syns in self.syns.values():
            for nq in syns["con"]:
                nq.clear_events()

    def config_DSGC(self):
        self.config_soma()
        self.config_dends()
        self.create_synapses()

        # pre-calculations based on set properties
        self.hard_offsets = {
            trans: [
                self.dir_sigmoids["offset"](d, p["pref_offset"], p["null_offset"])
                for d in self.dir_labels
            ]
            for trans, p in self.synprops.items()
            if trans != "PLEX"
        }

    def build_sac_net(self, rho=None):
        if rho is not None:
            self.sac_rho = rho

        probs = {
            t: {
                "null": self.synprops[t]["null_prob"],
                "pref": self.synprops[t]["pref_prob"],
            }
            for t in ["E", "I"]
        }

        self.sac_net: Optional[SacNetwork] = SacNetwork(
            self.syn_locs[:, :2],
            probs,
            self.sac_rho,
            self.sac_uniform_dist,
            self.sac_shared_var,
            self.sac_theta_vars,
            self.sac_gaba_coverage,
            self.dir_labels,
            self.np_rng,
            offset=self.sac_offset,
            theta_mode=self.sac_theta_mode,
            cell_pref=0,
            n_plexus_ach=self.n_plexus_ach,
        )

    def flash_onsets(self):
        """
        Calculate onset times for each synapse randomly as though stimulus was
        a flash. Timing jitter is applied with pseudo-random number generators.
        """

        # store timings for return (fine to ignore return)
        onset_times = {trans: [] for trans in ["E", "I", "AMPA", "NMDA"]}
        rand_on = {trans: 0.0 for trans in ["E", "I", "AMPA", "NMDA"]}
        stim = {
            "type": "flash",
            "rhos": {"time": self.time_rho, "space": self.space_rho},
        }

        for s in range(self.n_syn):
            # normally distributed mean onset time for this site (jittered around)
            onset = self.np_rng.normal(self.flash_mean, self.flash_variance)

            for t in self.synprops.keys():
                rand_on[t] = self.np_rng.normal(0, 1)
                onset_times[t].append(rand_on[t])

            rand_on["E"] = rand_on["I"] * self.time_rho + (
                rand_on["E"] * np.sqrt(1 - self.time_rho**2)
            )

            successes = self.get_failures(s, stim)
            for t, props in self.synprops.items():
                for success in successes[t]:
                    if success:
                        self.syns[t]["con"][s].add_event(
                            onset + rand_on[t] * props["var"]
                        )

                    # add variable delay til next quanta (if there is one)
                    onset += self.np_rng.normal(
                        self.quanta_inter, self.quanta_inter_var
                    )

        return onset_times

    def bar_sweep(self, syn, dir_idx):
        """Return activation time for the given synapse based on the light bar
        config and the pre-synaptic setup (e.g. SAC network/hard-coded offsets)
        """
        bar = self.light_bar
        r = -self.dir_rads[dir_idx]
        x, y = rotate(self.origin, self.syn_locs[syn, 0], self.syn_locs[syn, 1], r)
        loc = {"x": x, "y": y}
        ax = "x" if bar["x_motion"] else "y"

        on_times = {}

        # distance to synapse divided by speed
        for t in self.synprops.keys():
            if t in ["E", "I", "PLEX"] and self.sac_net is not None:
                sac_loc = self.sac_net.get_syn_loc(t, syn, r)
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

    def bar_onsets(self, stim):
        """
        Calculate onset times for each synapse based on when the simulated bar
        would be passing over their location, modified by spatial offsets.
        Timing jitter is applied using pseudo-random number generators.
        """
        sac = self.sac_net

        time_rho = stim.get("rhos", {"time": self.time_rho})["time"]
        rand_on = {trans: 0.0 for trans in ["E", "I", "AMPA", "NMDA"]}

        for s in range(self.n_syn):
            bar_times = self.bar_sweep(s, stim["dir"])
            if sac is None or not self.sac_angle_rho_mode or not sac.gaba_here[s]:
                syn_rho = time_rho
            else:
                # scale correlation of E and I by diff of their dend angles
                syn_rho = time_rho - time_rho * sac.deltas[s] / 180

            # shared jitter
            jit = self.np_rng.normal(0, 1)
            bar_times = {k: v + jit * self.jitter for k, v in bar_times.items()}

            for t in self.synprops.keys():
                rand_on[t] = self.np_rng.normal(loc=0.0, scale=1.0)

            rand_on["E"] = rand_on["I"] * syn_rho + (
                rand_on["E"] * np.sqrt(1 - syn_rho**2)
            )

            successes = self.get_failures(s, stim)
            for t, props in self.synprops.items():
                times = bar_times[t] if t == "PLEX" else [bar_times[t]]
                succs = successes[t] if t == "PLEX" else [successes[t]]
                rons = (
                    [rand_on[t]]
                    if t != "PLEX"
                    else self.np_rng.normal(loc=0.0, scale=1.0, size=sac.n_plexus_ach)
                )
                for tm, ss, rn in zip(times, succs, rons):
                    q_delay = 0.0
                    for success in ss:
                        if success:
                            self.syns[t if t != "PLEX" else "E"]["con"][s].add_event(
                                tm + q_delay + rn * props["var"]
                            )
                        # add variable delay til next quanta
                        q_delay += self.np_rng.normal(
                            self.quanta_inter, self.quanta_inter_var
                        )

    def bar_poissons(self, stim):
        sac = self.sac_net

        time_rho = stim.get("rhos", {"time": self.time_rho})["time"]

        for s in range(self.n_syn):
            bar_times = self.bar_sweep(s, stim["dir"])
            if sac is None or not self.sac_angle_rho_mode or not sac.gaba_here[s]:
                syn_rho = time_rho
            else:
                # scale correlation of E and I by diff of their dend angles
                syn_rho = time_rho - time_rho * sac.deltas[s] / 180

            probs = {}
            for t, props in self.synprops.items():
                if stim["type"] == "flash":
                    probs[t] = props["prob"]
                elif sac is not None and t in ["E", "I", "PLEX"]:
                    probs[t] = sac.probs[t][s, stim["dir"]]
                else:
                    # calculate probability of release
                    probs[t] = self.dir_sigmoids["prob"](
                        self.dirs[stim["dir"]], props["pref_prob"], props["null_prob"]
                    )

            poissons = {}

            if sac.gaba_here[s]:
                base_poisson = poisson_of_release(self.np_rng, self.sac_rate * syn_rho)
                for t in ["E", "I"]:
                    poissons[t] = [
                        np.round(
                            base_poisson
                            + poisson_of_release(
                                self.np_rng, self.sac_rate * (1 - syn_rho)
                            )
                            * probs[t]
                        ).astype(np.int)
                    ]
            else:
                poissons["E"] = [
                    poisson_of_release(self.np_rng, self.sac_rate * probs["E"])
                ]
                poissons["I"] = [np.array([0])]

            for t in ["AMPA", "NMDA"]:
                poissons[t] = [
                    poisson_of_release(self.np_rng, self.glut_rate * probs[t])
                ]

            if self.n_plexus_ach > 0:
                poissons["PLEX"] = [
                    poisson_of_release(self.np_rng, self.sac_rate * pr)
                    for pr in probs["PLEX"]
                ]

            for t in self.synprops.keys():
                times = bar_times[t] if t == "PLEX" else [bar_times[t]]
                for tm, psn in zip(times, poissons[t]):
                    qs = quanta_to_times(psn, self.dt) * 1000 + tm
                    t = t if t != "PLEX" else "E"
                    for q in qs:
                        self.syns[t]["con"][s].add_event(q)


    def get_failures(self, idx, stim):
        """
        Determine number of quantal activations of each synapse occur on a
        trial. Psuedo-random numbers generated for each synapse are compared
        against thresholds set by probability of release to determine if the
        "pre-synapse" succeeds or fails to release neurotransmitter.
        """
        sac = self.sac_net

        # numbers above can result in NaNs
        rho = stim.get("rhos", {"space": self.space_rho})["space"]
        rho = 0.986 if rho > 0.986 else rho

        # calculate input rho required to achieve the desired output rho
        # exponential fit: y = y0 + A * exp(-invTau * x)
        # y0 = 1.0461; A = -0.93514; invTau = 3.0506
        rho = 1.0461 - 0.93514 * np.exp(-3.0506 * rho)

        picks = {
            t: self.np_rng.normal(
                loc=0.0, scale=1.0, size=(None if t != "PLEX" else sac.n_plexus_ach)
            )
            for t in self.synprops.keys()
        }

        # correlate synaptic variance of ACH with GABA
        if sac is None or not self.sac_angle_rho_mode:
            picks["E"] = picks["I"] * rho + (picks["E"] * np.sqrt(1 - rho**2))
        else:
            # scale correlation of E and I by prox of their dend angles
            if sac.gaba_here[idx]:
                syn_rho = rho - rho * sac.deltas[idx] / 180
                picks["E"] = picks["I"] * syn_rho + picks["E"] * np.sqrt(
                    1 - syn_rho**2
                )

        probs = {}
        for t, props in self.synprops.items():
            if stim["type"] == "flash":
                probs[t] = props["prob"]
            elif sac is not None and t in ["E", "I", "PLEX"]:
                probs[t] = sac.probs[t][idx, stim["dir"]]
            else:
                # calculate probability of release
                probs[t] = self.dir_sigmoids["prob"](
                    self.dirs[stim["dir"]], props["pref_prob"], props["null_prob"]
                )

        # sdevs = {k: np.std(v) for k, v in picks.items()}
        # TODO: this adjustment should be flat since it is just happening for a single
        # synapse, but the sdev should still be 1 though, since the distribution shouldn't
        # be scaled by the correlation operation
        sdevs = {k: 1.0 for k in picks.keys()}
        successes = {}

        for t in picks.keys():
            # determine release bool for each possible quanta
            q_probs = np.array(
                [probs[t] * (self.quanta_Pr_decay**q) for q in range(self.max_quanta)]
            )
            q_probs = q_probs.T if t == "PLEX" else q_probs  # quanta to last dimension
            left = st.norm.ppf((1 - q_probs) / 2.0) * sdevs[t]
            right = st.norm.ppf(1 - (1 - q_probs) / 2.0) * sdevs[t]
            # an array of success boolean with length max quanta
            if t != "PLEX":
                successes[t] = (left < picks[t]) * (picks[t] < right)
            else:
                successes[t] = (left < picks[t].reshape(-1, 1)) * (
                    picks[t].reshape(-1, 1) < right
                )

        return successes

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
        locs = []
        per = self.rec_per_sec
        for dend in self.all_dends:
            dend.push()
            pts = int(h.n3d())

            # Rough coordinates based on indexing the list of 3d points
            # assigned to the current section. Number of points vary.
            for s in range(per):
                locs.append([h.x3d(s * (pts - 1) / per), h.y3d(s * (pts - 1) / per)])

            h.pop_section()

        return np.array(locs)

    def n_terminals(self):
        return len(self.terminals)
