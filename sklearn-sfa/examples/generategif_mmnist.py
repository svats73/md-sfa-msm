import numpy as np
from sksfa import SFA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


data = np.load("data/mmnist_data.npy").squeeze()
print(data.shape)
n_points = data.shape[0]

fig, ax = plt.subplots()
fig.set_size_inches(2.5, 2.5, True) 
image = ax.imshow(data[0].T, cmap="Greys")

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

def init():
    return image,

def update(frame):
    image.set_data(data[frame].T)
    return image,
plt.tight_layout()

ani = FuncAnimation(fig, update, frames=300, interval=50,
                    init_func=init, blit=True)

ani.save("../doc/images/moving_mnist.gif", writer="imagemagick", dpi=40)
