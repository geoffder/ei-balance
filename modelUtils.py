from neuron import h
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from general_utils import project_onto_line


def frange(start, stop, step):
    """iterator for floats"""
    i = start
    while i < stop:
        # makes this work as an iterator, returns value then continues
        # loop for next call of function
        yield i
        i += step


def find_origin(dends):
    """find the centre point of the cell/arbour"""
    leftX = 1000
    rightX = -1000
    topY = -1000
    botY = 1000
    for i in range(len(dends)):
        dends[i].push()

        # mid point
        pts = int(h.n3d())  # number of 3d points of section
        if pts % 2:  # odd number of points
            xLoc = h.x3d((pts - 1) / 2)
            yLoc = h.y3d((pts - 1) / 2)
        else:
            xLoc = (h.x3d(pts / 2) + h.x3d((pts / 2) - 1)) / 2.0
            yLoc = (h.y3d(pts / 2) + h.y3d((pts / 2) - 1)) / 2.0
        if xLoc < leftX:
            leftX = xLoc
        if xLoc > rightX:
            rightX = xLoc
        if yLoc < botY:
            botY = yLoc
        if yLoc > topY:
            topY = yLoc

        # terminal point
        xLoc = h.x3d(pts - 1)
        yLoc = h.y3d(pts - 1)
        if xLoc < leftX:
            leftX = xLoc
        if xLoc > rightX:
            rightX = xLoc
        if yLoc < botY:
            botY = yLoc
        if yLoc > topY:
            topY = yLoc

        h.pop_section()
    # print  'leftX: '+str(leftX)+ ', rightX: '+str(rightX)
    # print  'topY: '+str(topY)+ ', botY: '+str(botY)
    return (leftX + (rightX - leftX) / 2, botY + (topY - botY) / 2)


def rotate(origin, X, Y, angle):
    """
    Rotate a point (X[i],Y[i]) counterclockwise an angle around an origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    X, Y = np.array(X), np.array(Y)
    rotX = ox + np.cos(angle) * (X - ox) - np.sin(angle) * (Y - oy)
    rotY = oy + np.sin(angle) * (X - ox) + np.cos(angle) * (Y - oy)
    return rotX, rotY


def rot(angle, pos, about=np.zeros(2)):
    """
    Rotate a point pos counterclockwise an angle around an origin.
    The angle should be given in radians.
    """
    ox, oy = about[0], about[1]
    x, y = pos[0], pos[1]
    c = np.cos(angle)
    s = np.sin(angle)
    rot_x = ox + c * (pos[0] - ox) - s * (pos[1] - oy)
    rot_y = oy + s * (pos[0] - ox) + c * (pos[1] - oy)
    return np.array([rot_x, rot_y])


def find_spikes(Vm, thresh=20):
    """use scipy.signal.find_peaks to get spike count and times"""
    spikes, _ = find_peaks(Vm, height=thresh)  # returns indices
    count = spikes.size
    times = spikes * h.dt

    return count, times


def nrn_section(name):
    """Create NEURON hoc section, and return a corresponding python object."""
    h("create " + name)
    return h.__getattribute__(name)


def nrn_objref(name):
    """Create NEURON hoc objref, and return a corresponding python object."""
    h("objref " + name)
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


def build_stim():
    """Return basic initialized NetStim object."""
    stim = h.NetStim(0.5)
    stim.interval = 0
    stim.number = 1
    stim.noise = 0
    return stim


def windows_gui_fix():
    """Workaround GUI bug in windows preventing graphs drawing during run"""
    h(
        """
        proc advance() {
            fadvance()
            nrnpython("")
        }
    """
    )


def dist_calc(pth, model):
    """
    Get the location of each synapse or recording location and the cable
    distances (running along dendrites) between each of them.
    """
    sectionList = model.all_dends
    numLocs = len(model.all_dends) * model.recs_per_sec

    locs = {"X": [], "Y": []}
    dendNums = []
    for i in range(len(sectionList)):
        sectionList[i].push()
        pts = int(h.n3d())  # number of 3d points of section
        for s in range(model.recs_per_sec):
            if pts % 2:  # odd number of points
                locs["X"].append(h.x3d(s * (pts - 1) / (model.recs_per_sec)))
                locs["Y"].append(h.y3d(s * (pts - 1) / (model.recs_per_sec)))
            else:
                locs["X"].append(
                    (
                        h.x3d(s * (pts - 1) / (model.recs_per_sec))
                        + h.x3d(s * (pts - 1) / (model.recs_per_sec))
                        - 1
                    )
                    / 2.0
                )
                locs["Y"].append(
                    (
                        h.y3d(s * (pts - 1) / (model.recs_per_sec))
                        + h.y3d(s * (pts - 1) / (model.recs_per_sec))
                        - 1
                    )
                    / 2.0
                )

        # also get the name so I can make a lookup of pyIndex# to dend#
        name = h.secname()
        dendNums.append(name.replace("DSGC[0].dend[", "").replace("]", ""))
        h.pop_section()

    # print coordinates of all terminal dendrites to file
    locfname = "recLocations.csv"
    dendfname = "dendNumbers.csv"
    # x, y coordinates for each recording location
    locations = pd.DataFrame(locs)
    locations.to_csv(pth + locfname, index=False)
    # python index and neuron dend numbers
    dendNumbers = pd.DataFrame(dendNums, columns=["dendNum"])
    dendNumbers.to_csv(pth + dendfname, index_label="pyIdx")

    distBetwRecs = list(range(numLocs))
    for i in range(numLocs):
        distBetwRecs[i] = np.zeros(numLocs)
    iCnt = 0
    for i in range(len(sectionList)):
        sectionList[i].push()
        for iSeg in range(model.recs_per_sec):
            h.distance(0, iSeg * model.seg_step)  # set origin current seg
            kCnt = 0
            for k in range(len(model.all_dends)):
                sectionList[k].push()
                for kSeg in range(model.recs_per_sec):
                    distBetwRecs[iCnt][kCnt] = h.distance(kSeg * model.seg_step)
                    kCnt += 1
                h.pop_section()
            iCnt += 1
        h.pop_section()

        dists = pd.DataFrame(distBetwRecs)
        dists.to_csv(pth + "distBetwRecs.csv", index=False)


def cable_dist_to_soma(model):
    model.soma.push()
    h.distance(0, 0.5)  # set origin at soma
    dists = []
    for dend in model.all_dends:
        dend.push()
        dists.append(h.distance(1.0))
        h.pop_section()
    return dists


def map_tree(cell):
    """Sort dendrite sections by branch order and if they are terminal. Stack
    based non-recursive solution."""
    cell.soma.push()

    order_pos = [0]  # last item indicates which # child to check next
    order_list = [[]]  # list of lists, dends sorted by branch order
    terminals = []  # terminal branches (no children)
    non_terms = []  # non-terminal branches (with children)

    while True:
        sref = h.SectionRef()  # reference to current section

        # more children to check
        if order_pos[-1] < sref.nchild():
            if len(order_pos) > 1:  # exclude primes from non-terminals
                non_terms.append(h.cas())  # add this parent dend to list

            # access child of current
            sref.child[order_pos[-1]].push()

            if len(order_pos) > len(order_list):
                order_list.append([])

            order_list[len(order_pos) - 1].append(h.cas())  # order child
            order_pos.append(0)  # extend to next order
        else:
            if len(order_pos) == 1:
                # exceeded number of prime dends
                break  # entire tree is mapped
            else:
                if not sref.nchild():
                    # add childless dend to list
                    terminals.append(h.cas())

                # back an order, target next child there.
                del order_pos[-1]
                order_pos[-1] += 1
                h.pop_section()

    return order_list, terminals, non_terms


def merge(old, new):
    """Recursively merge nested dictionaries. Used to updating parameter
    dictionaries."""
    for k, v in new.items():
        if type(v) is dict:
            old[k] = merge(old[k], v)
        else:
            old[k] = v

    return old


def wrap_360(theta):
    """Wrap degrees from -180 -> 180 scale and super 360 to 0 -> 360 scale."""
    if np.isnan(theta):
        return theta
    else:
        theta = theta % 360
        if theta < 0:
            return theta + 360
        else:
            return theta


def wrap_180(theta):
    """Wrap degrees from 360 scale (and super 180) to 0 -> 180 scale."""
    if np.isnan(theta):
        return theta
    else:
        theta = theta % 360
        if theta > 180:
            return 360 - theta
        else:
            return theta


def scale_180_from_360(theta):
    """Wrap degrees from 360 scale to -180 -> 180 scale."""
    # return (((theta - 180) % 360) - 180)
    return (((theta - 180) % 360) - 180) if theta != 180 else 180

def calc_sweep_times(origin, dir_rads, bar, syn_locs, sac_net, hard_offsets, syn_kinds=["E", "I", "NMDA"]):
    """Return dict with synapse activation times for each direction.
    dir_idx -> syn_idx -> syn_kind"""
    times = {}  # dir -> syn -> trans
    start = bar["start_time"]
    v = bar["speed"]
    for i, r in enumerate(dir_rads):  # type:ignore
        times[i] = {}
        line_a = rot(r, bar["rel_start_pos"]) + origin
        line_b = rot(r, bar["rel_end_pos"]) + origin

        for syn in range(syn_locs.shape[0]):
            times[i][syn] = {}
            syn_dist = project_onto_line(line_a, line_b, syn_locs[syn, :2])[0]
            for t in syn_kinds:
                if t in ["E", "I", "PLEX"] and sac_net is not None:
                    sac_loc = sac_net.get_syn_loc(t, syn, 0)
                    if t == "PLEX":
                        locs = np.array([sac_loc["x"], sac_loc["y"]]).T
                        dist = np.array(
                            [project_onto_line(line_a, line_b, l)[0] for l in locs]
                        )
                    else:
                        sac_loc = np.array([sac_loc["x"], sac_loc["y"]])
                        dist = project_onto_line(line_a, line_b, sac_loc)[0]
                    times[i][syn][t] = start + dist / v
                else:
                    offset = hard_offsets[t][i]
                    times[i][syn][t] = start + (offset + syn_dist) / v
    return times
