# -*- coding: utf-8 -*-
# -*- mode: python -*-
# Adapted from mpl_toolkits.axes_grid1
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)
# gist created by dmeliza (https://gist.github.com/dmeliza/3251476)
# TODO: adapt my own

from matplotlib.offsetbox import AnchoredOffsetbox


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(
        self,
        transform,
        sizex=0,
        sizey=0,
        labelx=None,
        labely=None,
        pad=0,
        loc=4,
        borderpad=0.1,
        sep=2,
        xsep=None,
        ysep=None,
        prop=None,
        barcolor="black",
        barwidth=1.,
        textprops={"fontsize": 13},
            xtextprops={},
            ytextprops={},
        **kwargs
    ):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import (
            AuxTransformBox,
            VPacker,
            HPacker,
            TextArea,
            DrawingArea,
        )

        inv = transform.inverted()  # convert display coords to axis coords
        bars = AuxTransformBox(transform)
        xbar = AuxTransformBox(transform)
        ybar = AuxTransformBox(transform)
        bar_off = (barwidth - 1)  / -2 - 1  # fudging to get butts to line up
        x_pos = inv.transform((bar_off, 0) if (sizex and sizey) else (0, 0))
        y_pos = inv.transform((0, bar_off) if (sizex and sizey) else (0, 0))

        if sizex:
            bars.add_artist(
                Rectangle(x_pos, sizex, 0, ec=barcolor, lw=barwidth, fc="none")
            )
        if sizey:
            bars.add_artist(
                Rectangle(y_pos, 0, sizey, ec=barcolor, lw=barwidth, fc="none")
            )

        if sizex and labelx:
            self.xlabel = TextArea(labelx, textprops={**textprops, **xtextprops})
            xsep = sep if xsep is None else xsep
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=xsep)
        if sizey and labely:
            self.ylabel = TextArea(labely, textprops={**textprops, **ytextprops})
            ysep = sep if ysep is None else ysep
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=ysep)

        AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=bars,
            prop=prop,
            frameon=False,
            **kwargs
        )


def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """

    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    if matchx:
        kwargs["sizex"] = f(ax.xaxis)
        kwargs["labelx"] = str(kwargs["sizex"])
    if matchy:
        kwargs["sizey"] = f(ax.yaxis)
        kwargs["labely"] = str(kwargs["sizey"])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)
    if hidex and hidey:
        ax.set_frame_on(False)

    return sb
