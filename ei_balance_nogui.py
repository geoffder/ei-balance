from ei_balance import *

h.load_file("stdgui.hoc")  # headless, still sets up environment
h.cvode.cache_efficient(1)
from neuron import coreneuron
coreneuron.enable = True
