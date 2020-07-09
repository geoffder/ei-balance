import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wrap(angle):
    angle %= 360
    if angle <= -180:
        return angle + 360
    elif angle > 180:
        return angle - 360
    else:
        return angle

if __name__ == "__main__":
    basest = "/mnt/Data/NEURONoutput/"
    df = pd.read_csv(basest + "briggman_approx_angle_counts.csv")

    prefs = df["cell_pref"]
    bin_ax = np.array(df.columns[1:], dtype=np.int)
    bin_counts = df.values[:, 1:]

    wrapper = np.vectorize(wrap)
    rel_null = wrapper([bin_ax - wrap(p + 180) for p in prefs])

    order = np.arange(-150, 181, 30)
    sort_idxs = np.argsort(rel_null, axis=1)
    for i in range(sort_idxs.shape[0]):
        bin_counts[i] = bin_counts[i, sort_idxs[i]]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(order, bin_counts.T)
    ax[0].set_ylabel("Synapse Count")
    ax[1].plot(order, bin_counts.mean(axis=0))
    ax[1].set_ylabel("Synapse Count (Average)")
    ax[1].set_xlabel("Angle (° relative to null)")

    fig.tight_layout()
    plt.show()
