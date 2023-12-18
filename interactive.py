import numpy as np
import analysis as ana
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider
from matplotlib.patches import Rectangle
from modelUtils import rotate


class MotionResponse:
    def __init__(
        self,
        net_data,
        tree_k="Vm",
        dsgc_alpha=0.35,
        delta=1,
        figsize=(5, 6),
        tree_vmin=None,
        tree_vmax=None,
        soma_vmin=-70,
        soma_vmax=30,
        bar_width=200,
        bar_height=300,
        bar_edge=(0, 0, 0, 0.5),
        bar_face=(0, 0, 0, 0.0),
        bar_hatch=None,
        bar_lw=2,
    ):
        self.delta = delta
        self.soma = net_data["soma"]["Vm"]
        self.tree_k = tree_k
        self.tree = net_data["dendrites"][tree_k]
        self.dirs = net_data["params"]["dir_labels"]

        self.n_trials = self.soma.shape[0]
        self.soma_pts = self.soma.shape[-1]
        self.tree_pts = self.tree.shape[-1]

        self.build_fig(figsize)

        dsgc_img, extent = ana.get_dsgc_img()
        self.tree_ax.imshow(dsgc_img, alpha=dsgc_alpha, extent=extent, cmap="gray")
        self.tree_ax.set_xlim(-30, 230)
        self.tree_ax.set_ylim(-30, 260)

        self.rate_diff = self.soma_pts // self.tree_pts
        self.tree_t = 0
        locs = net_data["dendrites"]["locs"]
        vmin = np.min(self.tree) if tree_vmin is None else tree_vmin
        vmax = np.max(self.tree) if tree_vmax is None else tree_vmax
        self.scat = self.tree_ax.scatter(
            locs[:, 0], locs[:, 1], c=self.tree[0, 0, :, 0], vmin=vmin, vmax=vmax
        )

        self.sweeps = self.bar_corner_pos(net_data["params"], bar_width, bar_height)
        self.stim = Rectangle(
            self.sweeps[0][0],
            bar_width,
            bar_height,
            lw=bar_lw,
            fc=bar_face,
            ec=bar_edge,
            angle=self.dirs[0],
            hatch=bar_hatch,
        )
        self.tree_ax.add_patch(self.stim)
        self.colorbar = self.fig.colorbar(
            self.scat, cax=self.colorbar_ax, orientation="vertical"
        )

        self.time: np.ndarray = np.linspace(
            0, self.soma_pts * net_data["params"]["dt"], self.soma_pts
        )
        self.line = self.soma_ax.plot(self.time, self.soma[0, 0])[0]
        self.t_mark = self.soma_ax.plot(
            self.time[0], self.soma[0, 0, 0], marker="x", c="red"
        )[0]
        self.soma_ax.set_xlim(self.time.min(), self.time.max())
        self.soma_ax.set_ylim(soma_vmin, soma_vmax)

        self.connect_events()

        self.update()

    def build_fig(self, figsize):
        self.fig = plt.figure(constrained_layout=True, figsize=figsize)
        ncols = 10
        edge_w = 0.075
        mid_w = (1 - edge_w * 2) / (ncols - 2)
        gs = self.fig.add_gridspec(
            nrows=3,
            ncols=ncols,
            width_ratios=[edge_w] + (ncols - 2) * [mid_w] + [edge_w],
            height_ratios=[0.75, 0.2, 0.05],
        )
        self.tree_ax = self.fig.add_subplot(gs[0, 1:9])
        self.colorbar_ax = self.fig.add_subplot(gs[0, 9])
        self.soma_ax = self.fig.add_subplot(gs[1, :])
        self.soma_ax.set_xlabel("Time (ms)")
        self.delta_ax = self.fig.add_subplot(gs[2, 0:3])
        self.delta_box = TextBox(self.delta_ax, "", initial=str(self.delta))
        self.delta_ax.set_title("delta (scroll)")
        self.trial_slide_ax = self.fig.add_subplot(gs[2, 3:6])
        self.build_trial_slide_ax()
        self.dir_slide_ax = self.fig.add_subplot(gs[2, 6:])
        self.build_dir_slide_ax()

    def bar_corner_pos(self, ps, bar_width, bar_height):
        bar_start = np.array([ps["light_bar"]["x_start"], ps["origin"][1]])
        net_origin = ps["origin"]
        step = ps["dt"] * ps["light_bar"]["speed"]
        n_pts = self.soma_pts

        x0 = bar_start[0] - bar_width
        y0 = bar_start[1] - bar_height / 2
        xs = np.array([x0 + step * t for t in range(n_pts)])
        ys = np.array([y0 for _ in range(n_pts)])

        sweeps = {}
        for i in range(len(self.dirs)):
            theta = np.radians(self.dirs[i])
            dir_xs, dir_ys = rotate(net_origin, xs, ys, theta)
            sweeps[i] = np.array([dir_xs, dir_ys]).T

        return sweeps

    def build_trial_slide_ax(self):
        vmax = self.n_trials - 1
        self.trial_idx = 0
        self.trial_slider = Slider(
            self.trial_slide_ax,
            "",
            valmin=0,
            valmax=vmax if vmax > 0 else 0.1,
            valinit=0,
            valstep=1,
        )
        self.trial_slider.valtext.set_visible(False)
        self.trial_slide_ax.set_title("trial = %i" % 0)

    def on_trial_slide(self, v):
        self.trial_idx = int(v)
        self.update()
        self.trial_slide_ax.set_title("trial = %i" % self.trial_idx)

    def build_dir_slide_ax(self):
        self.dir_idx = 0
        self.dir_slider = Slider(
            self.dir_slide_ax,
            "",
            valmin=0,
            valmax=(len(self.dirs) - 1),
            valinit=0.1,
            valstep=1,
        )
        self.dir_slider.valtext.set_visible(False)
        self.dir_slide_ax.set_title("dir = %i" % self.dirs[0])

    def set_delta(self, s):
        try:
            d = int(s)
            if d < 1:
                raise ValueError("delta must be >= 1")
            self.delta = d
        except:
            self.delta_box.set_val(str(self.delta))

    def on_dir_slide(self, v):
        self.dir_idx = int(v)
        self.dir_slide_ax.set_title("dir = %i" % self.dirs[self.dir_idx])
        self.stim.set_angle(self.dirs[self.dir_idx])
        self.update()

    def on_scroll(self, event):
        if event.button == "up":
            if event.inaxes == self.delta_ax:
                self.delta = self.delta + 1
                self.delta_box.set_val(str(self.delta))
            else:
                self.tree_t = (self.tree_t + self.delta) % self.tree_pts
        else:
            if event.inaxes == self.delta_ax and self.delta > 1:
                self.delta = self.delta - 1
                self.delta_box.set_val(str(self.delta))
            else:
                self.tree_t = (self.tree_t - self.delta) % self.tree_pts
        self.update()

    def on_soma_click(self, event):
        if (
            event.button == 1
            and event.inaxes == self.soma_ax
            and self.fig.canvas.manager.toolbar.mode == ""  # ignore if panning/zooming
        ):
            self.tree_t = ana.nearest_index(self.time, event.xdata) // self.rate_diff
            self.update()

    def update(self):
        soma_t = self.tree_t * self.rate_diff
        self.scat.set_array(self.tree[self.trial_idx, self.dir_idx, :, self.tree_t])
        self.line.set_ydata(self.soma[self.trial_idx, self.dir_idx])
        self.t_mark.set_data(
            self.time[soma_t], self.soma[self.trial_idx, self.dir_idx, soma_t]
        )
        self.stim.set_xy(self.sweeps[self.dir_idx][soma_t])
        self.scat.axes.figure.canvas.draw()

    def connect_events(self):
        self.fig.canvas.capture_scroll = True
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.trial_slider.on_changed(self.on_trial_slide)
        self.dir_slider.on_changed(self.on_dir_slide)
        self.fig.canvas.mpl_connect("button_release_event", self.on_soma_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_soma_click)
        self.delta_box.on_submit(self.set_delta)
