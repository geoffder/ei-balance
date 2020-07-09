import numpy as np
import matplotlib.pyplot as plt

length = 10
degs = np.linspace(0, 315, 8)
rads = np.radians(degs)
xs = np.array([length*np.cos(rad) for rad in rads])
ys = np.array([length*np.sin(rad) for rad in rads])

for i in range(xs.shape[0]):
    plt.plot([0, xs[i]], [0, ys[i]], label=degs[i])
plt.ylabel('Y')
plt.xlabel('X')
plt.legend()
plt.show()
