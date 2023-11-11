from neuron import h
import numpy as np

# import numba
# from typing import List


class NetQuanta:
    def __init__(self, syn, weight, delay=0.0):
        self.con = h.NetCon(None, syn, 0.0, 0.0, weight)
        self.delay = delay  # NetCon does not impose delay on *scheduled* events
        self._events = []

    @property
    def weight(self):
        return self.con.weight[0]

    @property
    def events(self):
        return self._events

    @weight.setter
    def weight(self, w):
        self.con.weight[0] = w

    @events.setter
    def events(self, ts):
        self._events = [t + self.delay for t in ts]

    def clear_events(self):
        self._events = []

    def add_event(self, t):
        self._events.append(t + self.delay)

    # @staticmethod
    # @numba.jit("void(List(float64), float64, int64[:], float64, float64)")
    # def _add_quanta(
    #     _events: List[float], delay: float, quanta: np.ndarray, dt: float, t0: float
    # ):
    #     i: int = len(_events)
    #     j: int
    #     _events += [0.0] * np.sum(quanta)  # type:ignore
    #     t: float = t0 + delay
    #     for j in range(quanta.shape[0]):
    #         for _ in range(quanta[j]):
    #             _events[i] = t  # type:ignore
    #             i += 1
    #         t += dt

    # def add_quanta(self, quanta: np.ndarray, dt: float, t0: float):
    #     self._add_quanta(self._events, self.delay, quanta, dt, t0)

    def add_quanta(self, quanta, dt, t0, jitters=None):
        i = len(self._events)
        self._events += [None] * max(0, np.sum(quanta))
        jitters = np.zeros(len(quanta)) if jitters is None else jitters
        t = t0 + self.delay
        for n, j in zip(quanta, jitters):
            jittered = t + j
            for _ in range(n):
                self._events[i] = jittered  # type:ignore
                i += 1
            t += dt

    def initialize(self):
        """Schedule events in the NetCon object. This must be called within the
        function given to h.FInitializeHandler in the model running class/functions.
        """
        for ev in self._events:
            self.con.event(ev)
