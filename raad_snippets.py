from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt


def nrn_section(name):
    """Create NEURON hoc section, and return a corresponding python object."""
    h("create " + name)
    return h.__getattribute__(name)


class NrnObjref:
    """Creates a hoc objref, and wraps it in a python object with setting/getting
    helper functions. Initializing objects in this way will allow use to easily
    access them by `name` in the nrngui."""
    def __init__(self, name, n=1):
        self.name = name
        self.n = 1
        h("objref %s[%i]" % (name, n))
        self.ref = h.__getattribute__(name) 

    def __len__(self):
        return self.n
    
    def __getitem__(self, i):
        return self.ref[i]

    def __setitem__(self, i, obj):
        self.ref[i] = obj

    def __iter__(self):
        return iter(self.ref)
    
    def set(self, obj, i=0):
        """Assign to given hoc object to index `i` in the objref array that
        backs this python object.
        """
        self.ref[i] = obj


class Synapse:
    def __init__(self, name, section, **config):
        self.syn  = NrnObjref("%s_syn" % name)
        self.stim = NrnObjref("%s_stim" % name)
        self.conn = NrnObjref("%s_conn" % name)

        section.push()  # objects will be placed into the currently accessed section
        self.syn.set(self.build_syn(config))
        self.stim.set(self.build_stim(config))
        self.conn.set(self.build_conn(
            self.stim[0], self.syn[0], 0, 0, config.get("weight", .00025)))
        h.pop_section()

    @staticmethod
    def build_syn(config):
        """Return basic initialized Exp2Syn object."""
        syn      = h.Exp2Syn(config.get("pos", .5))
        syn.tau1 = config.get("tau1", 0)  # rise exponential
        syn.tau2 = config.get("tau1", 4)  # decay exponential
        syn.e    = config.get("e", 0)     # reversal potential
        return syn
        
    @staticmethod
    def build_stim(config):
        """Return basic initialized NetStim object. Drives a Syn object."""
        stim          = h.NetStim(config.get("pos", .5))
        stim.number   = config.get("n", 1)         # number of stims in train
        stim.start    = config.get("start", 100)   # train start time [ms]
        stim.interval = config.get("interval", 0)  # inter-stimulus interval
        stim.noise    = config.get("noise", 0)     # variation of isi
        return stim

    @staticmethod
    def build_conn(stim, syn, weight):
        """The NetCon object connects a NetStim to a Stim at a given weight.
        When the NetStim `stim` "fires", the `syn` will play out an event
        according to it's time constants and reversal potential. The conductance
        actually applied to the section `syn` resides in is determined by the
        weight of the NetCon.
        """
        return h.NetCon(stim, syn, 0, 0, weight)  # [weight in microsiemans]
        
    def set_start(self, t):
        "Set stimulus start time."
        self.stim[0].start = t
        
    def set_weight(self, w):
        "Set connection weight (conductance [microsiemans])"
        self.conn[0].weight[0] = w

    def set_reversal(self, e):
        "Set reversal potential of the `syn`."
        self.syn[0].e = e


def build_soma():
    soma = nrn_section("soma")

    soma.L    = 10  # [um]
    soma.diam = 10  # [um]
    soma.nseg = 1
    soma.Ra   = 100

    soma.insert("HHst")
    soma.gnabar_HHst = 0.15       # [S/cm2]
    soma.gkbar_HHst  = 0.035      # [S/cm2]
    soma.gkmbar_HHst = 0.003      # [S/cm2]
    soma.gleak_HHst  = 0.0001667  # [S/cm2]
    soma.eleak_HHst  = -65.0      # [mV]
    soma.NF_HHst     = 0.75

    return soma


def build_dend():
    dend = nrn_section("dend")

    dend.L    = 200   # [um]
    dend.diam = .75   # [um]
    dend.nseg = 3
    dend.Ra   = 100

    dend.insert("HHst")
    dend.gnabar_HHst = 0.03       # [S/cm2]
    dend.gkbar_HHst  = 0.025      # [S/cm2]
    dend.gkmbar_HHst = 0.003      # [S/cm2]
    dend.gleak_HHst  = 0.0001667  # [S/cm2]
    dend.eleak_HHst  = -65.0      # [mV]
    dend.NF_HHst     = 0.25
    return dend


def build_ball_and_stick():
    soma = build_soma()
    dend = build_dend()
    dend.connect(soma, 0.5, 0)
    
    return soma, dend


def h_init_run():
    h.init()
    h.run()


class Rig:
    def __init__(self, model):
        self.model = model       # dict of sections (using to update noise)
        self.nz_seed = 0         # used to update HH membrane noise
        self.electrodes = {}     # hoc vectors used to record membrane voltage
        self.recordings = {}     # stored membrane recordings
        self.voltage_clamps = {} # hoc SEClamp objects
        self.vc_electrodes = {}  # hoc vectors used to vc current
        self.vc_recordings = {}  # stored voltage clamp current recordings

    def place_electrode(self, name, section, pos):
        elec = h.Vector()                 # hoc vector object
        elec.record(section(pos)._ref_v)  # record voltage at `pos` in `section`
        self.electrodes[name] = elec
        self.recordings[name] = []

    def activate_vc_mode(self):
        """Set voltage sensitive conductances to zero."""
        for sec in self.model.values():
            sec.gnabar_HHst = 0.0
            sec.gkbar_HHst  = 0.0
            sec.gkmbar_HHst = 0.0
            sec.gleak_HHst  = 0.0
        
    def place_voltage_clamp(self, name, section, pos, **config):
        section.push()  # SEClamp is placed in currently accessed section
        vc = NrnObjref("VC")
        vc.set(h.SEClamp(pos))

        self.voltage_clamps[name] = vc
        self.config_voltage_clamp(name, **config)

        rec = h.Vector()
        rec.record(vc[0]._ref_i)  # record current
        self.vc_electrodes[name] = rec
        self.vc_recordings[name] = []

    def config_voltage_clamp(self, name, **config):
        vc = self.voltage_clamps[name]
        # default to same voltage throughout
        vc[0].dur1 = config.get("dur1", h.tstop)
        vc[0].dur2 = config.get("dur2", 0)
        vc[0].dur3 = config.get("dur3", 0)
        # default to initialization voltage
        vc[0].amp1 = config.get("amp1", h.v_init)
        vc[0].amp2 = config.get("amp2", h.v_init)
        vc[0].amp3 = config.get("amp3", h.v_init)
        
    def update_noise(self):
        for sec in self.model.values():
            sec.seed_HHst = self.nz_seed
            self.nz_seed += 1

    def run(self):
        for vec in self.electrodes.values():
            vec.resize(0)  # sets length of hoc vector to 0

        for vec in self.vc_electrodes.values():
            vec.resize(0) 

        self.update_noise()
        h_init_run()

        # copy and store contents of recording vectors into numpy arrays and
        # add them to our collections
        for name, vec in self.electrodes.items():
            self.recordings[name].append(np.array(vec))

        for name, vec in self.vc_electrodes.items():
            self.vc_recordings[name].append(np.array(vec))

    def clear_recordings(self):
        self.recordings = {name: [] for name in self.recordings.keys()}
        self.vc_recordings = {name: [] for name in self.vc_recordings.keys()}

    def get_recordings(self, name):
        return np.array(self.recordings[name])

    def get_vc_recordings(self, name):
        return np.array(self.vc_recordings[name])

    def plot_recordings(self):
        """Create a subplot for each of our recording locations."""
        fig, ax = plt.subplots(len(self.electrodes), 1, sharex=True)
        ax = np.array(ax) if not type(ax) == np.ndarray else ax

        timeax = np.arange(int(h.tstop * h.steps_per_ms)) * h.dt
        for a, name in zip(ax, self.recordings.keys()):
            a.plot(timeax, self.get_recordings(name)[:, :-1].T)
            a.set_ylabel("Voltage (mV)")
            a.set_title(name)

        ax[-1].set_xlabel("Time (ms)")
        plt.show()
        return fig
        
    def plot_vc_recordings(self):
        fig, ax = plt.subplots(len(self.vc_electrodes), 1, sharex=True)
        ax = np.array(ax) if not type(ax) == np.ndarray else ax

        timeax = np.arange(int(h.tstop * h.steps_per_ms)) * h.dt
        for a, name in zip(ax, self.vc_recordings.keys()):
            a.plot(timeax, self.get_vc_recordings(name)[:, :-1].T)
            a.set_ylabel("Voltage (mV)")
            a.set_title(name)

        ax[-1].set_xlabel("Time (ms)")
        plt.show()
        return fig


if __name__ == "__main__":
    soma, dend = build_ball_and_stick()
    exc_syn = Synapse("exc", dend, pos=1.0)
    inhib_syn = Synapse("inhib", dend, pos=0.5, tau1=2, tau2=10, e=-65, weight=.001)
    soma.push()

    h.tstop        = 200
    h.steps_per_ms = 10
    h.dt           = 0.1
    h.v_init       = -65
    h.celsius      = 36.9

    model = {"soma": soma, "dend": dend}
    rig = Rig(model)
    rig.place_electrode("soma", soma, 0.5)
    rig.place_electrode("dend", dend, 1.0)
    
    h.xopen("raad.ses")  # open neuron gui session
