import numpy as np
import matplotlib.colors as col
import hsluv

# TODO: give desired colour for hue as name or hex, using col.rgb_to_hsl to convert?
#
# adapted from https://stackoverflow.com/a/34557535 (credit: poppie)
def angle_cmap(steps=256, pos_hue=11.6, neg_hue=258.6, use_hpl=True):
    h = np.ones(steps)  # hue
    h[: steps // 2] = pos_hue  # red default (11.6)
    h[steps // 2 :] = neg_hue  # blue default (258.6)
    s = 100  # saturation
    l = np.linspace(0, 100, steps // 2)  # luminosity
    l = np.hstack((l, l[::-1]))

    colorlist = np.zeros((steps, 3))
    for ii in range(steps):
        if use_hpl:
            colorlist[ii, :] = hsluv.hpluv_to_rgb((h[ii], s, l[ii]))
        else:
            colorlist[ii, :] = hsluv.hsluv_to_rgb((h[ii], s, l[ii]))
    colorlist[colorlist > 1] = 1  # correct numeric errors
    colorlist[colorlist < 0] = 0
    return col.ListedColormap(colorlist)
