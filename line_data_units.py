import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# https://stackoverflow.com/questions/19394505/expand-the-line-with-specified-width-in-data-unit/42972469#42972469
class LineDataUnits(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72.0 / self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data)) - trans((0, 0))) * ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


if __name__ == "__main__":
    """
    line = LineDataUnits(x, y, linewidth=1, alpha=0.4)
    ax.add_line(line)

    ax.legend([Line2D([],[], linewidth=3, alpha=0.4)],
    ['some 1 data unit wide line'])    # <- legend possible via proxy artist
    plt.show()
    """
    pass
