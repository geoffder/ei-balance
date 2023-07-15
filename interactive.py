import numpy as np
import analysis as ana
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider


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
    ):
        self.delta = delta
        self.soma = net_data["soma"]["Vm"]
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
        # locs = net_data["dendrites"]["locs"] * px_per_um + np.array([x_off, y_off])
        locs = net_data["dendrites"]["locs"]
        vmin = np.min(self.tree) if tree_vmin is None else tree_vmin
        vmax = np.max(self.tree) if tree_vmax is None else tree_vmax
        self.scat = self.tree_ax.scatter(
            locs[:, 0], locs[:, 1], c=self.tree[0, 0, :, 0], vmin=vmin, vmax=vmax
        )

        self.init_bar_sweeps(net_data["params"])
        self.stim = self.tree_ax.plot(
            self.sweeps[0]["x"], self.sweeps[0]["y"], c="black", linewidth=3, alpha=0.5
        )[0]

        self.time: np.ndarray = np.linspace(
            0, self.soma_pts * net_data["params"]["dt"], self.soma_pts
        )
        self.line = self.soma_ax.plot(self.time, self.soma[0, 0])[0]
        self.t_mark = self.soma_ax.plot(
            self.time[0], self.soma[0, 0, 0], marker="x", c="red"
        )[0]
        self.soma_ax.set_xlim(self.time.min(), self.time.max())
        self.soma_ax.set_ylim(-70, 30)

        self.connect_events()

        self.update()

    def build_fig(self, figsize):
        self.fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = self.fig.add_gridspec(nrows=3, ncols=3, height_ratios=[0.75, 0.2, 0.05])
        self.tree_ax = self.fig.add_subplot(gs[0, :])
        self.soma_ax = self.fig.add_subplot(gs[1, :])
        self.soma_ax.set_xlabel("Time (s)")
        self.delta_ax = self.fig.add_subplot(gs[2, 0])
        self.delta_box = TextBox(self.delta_ax, "", initial=str(self.delta))
        self.delta_ax.set_title("delta (scroll)")
        self.trial_slide_ax = self.fig.add_subplot(gs[2, 1])
        self.build_trial_slide_ax()
        self.dir_slide_ax = self.fig.add_subplot(gs[2, 2])
        self.build_dir_slide_ax()

    def init_bar_sweeps(self, ps):
        bar_start = np.array([ps["light_bar"]["x_start"], ps["origin"][1]])
        self.sweeps = {}
        for i in range(len(ps["dir_labels"])):
            bar_x, bar_y = ana.generate_bar_sweep(
                {
                    "theta": ps["dir_labels"][i],
                    "net_origin": ps["origin"],
                    "start": bar_start,
                    "speed": ps["light_bar"]["speed"],
                    "dt": ps["dt"],
                    "n_pts": self.soma_pts,
                }
            )
            self.sweeps[i] = {"x": bar_x, "y": bar_y}

    def build_trial_slide_ax(self):
        self.trial_idx = 0
        self.trial_slider = Slider(
            self.trial_slide_ax,
            "",
            valmin=0,
            valmax=(self.n_trials - 1),
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
        if event.button == 1 and event.inaxes == self.soma_ax:
            self.tree_t = ana.nearest_index(self.time, event.xdata) // self.rate_diff
            self.update()

    def update(self):
        soma_t = self.tree_t * self.rate_diff
        self.scat.set_array(self.tree[self.trial_idx, self.dir_idx, :, self.tree_t])
        self.line.set_ydata(self.soma[self.trial_idx, self.dir_idx])
        self.t_mark.set_data(
            self.time[soma_t], self.soma[self.trial_idx, self.dir_idx, soma_t]
        )
        self.stim.set_xdata(self.sweeps[self.dir_idx]["x"][soma_t])
        self.stim.set_ydata(self.sweeps[self.dir_idx]["y"][soma_t])
        self.scat.axes.figure.canvas.draw()

    def connect_events(self):
        self.fig.canvas.capture_scroll = True
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.trial_slider.on_changed(self.on_trial_slide)
        self.dir_slider.on_changed(self.on_dir_slide)
        self.fig.canvas.mpl_connect("button_release_event", self.on_soma_click)
        self.delta_box.on_submit(self.set_delta)
