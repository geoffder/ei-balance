from neuron import h, gui
import numpy as np

def nrn_section(name):
    """Create NEURON hoc section, and return a corresponding python object."""
    h("create " + name)
    return h.__getattribute__(name)


class NrnObjref:
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
        self.ref[i] = obj

        
def nrn_objref(name, n=1):
    """Create NEURON hoc objref, and return a corresponding python object."""
    h("objref %s[%i]" % (name, n))
    return h.__getattribute__(name)


def build_stim(pos, n=1, start=100, interval=0, noise=0):
    """Return basic initialized NetStim object."""
    stim          = h.NetStim(pos)
    stim.number   = n
    stim.start    = start
    stim.interval = interval
    stim.noise    = noise
    return stim


def build_syn(pos, tau1, tau2, e):
    syn      = h.Exp2Syn(pos)
    syn.tau1 = tau1
    syn.tau2 = tau2
    syn.e    = e
    return syn
    

def build_synapse(name, section, pos=.5, tau1=0, tau2=4, e=0, weight=.00025):
    syn = {
        "syn": NrnObjref("%s_syn" % name),
        "stim": NrnObjref("%s_stim" % name),
        "conn": NrnObjref("%s_conn" % name)
    }
    section.push()
    syn["syn"].set(build_syn(pos, tau1, tau2, e))
    syn["stim"].set(build_stim(pos))
    syn["conn"].set(h.NetCon(syn["stim"][0], syn["syn"][0], 0, 0, weight))
    h.pop_section()
    return syn


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
    soma.NF_HHst     = 0.25

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

    
if __name__ == "__main__":
    soma, dend = build_ball_and_stick()
    syn = build_synapse("exc", dend, pos=1.0)
    soma.push()

    h.tstop        = 200
    h.steps_per_ms = 10
    h.dt           = 0.1
    h.v_init       = -65
    h.celsius      = 36.9

    h.xopen("raad.ses")  # open neuron gui session

