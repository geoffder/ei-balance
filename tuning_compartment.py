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
    build_stim,
    find_origin,
    rotate,
    merge,
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


class Cell:
    def __init__(self):
        self.soma = nrn_objref("soma")
        self.config_soma()

    def config_soma(self):
        """Build and set membrane properties of soma compartment"""
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
        

def set_hoc_params():
    """Set hoc NEURON environment model run parameters."""
    h.tstop = 100  # [ms]
    h.steps_per_ms = 10
    h.dt = 0.1  # [ms]
    h.v_init = -60  # [mV]
    h.celsius = 36.9


if __name__ == "__main__":
    plat = platform.system()
    if plat == "Linux":
        basest = "/mnt/Data/NEURONoutput/"
    else:
        basest = "D:\\NEURONoutput\\"
        windows_gui_fix()
