"""
============
Moving Digit
============
An example of linear :class:`sksfa.SFA` applied to a simple image time-series:
a one-digit version of the moving MNIST dataset. Each data point is 4096-dimensional.

.. image:: ../images/moving_mnist.gif
   :align: center

If the change in x is not significantly faster or slower than the change in y, linear SFA with only two output features
successfully extracts a smooth (and possibly flipped) representation of the position of the digit in the image.

Ground truth is only added for the comparison, not during training.
"""

import numpy as np
from sksfa import SFA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

split_ratio = 0.7
all_sequences = np.load("data/mmnist_data.npy").squeeze()
ground_truth = np.load("data/mmnist_positions.npy").squeeze()
n_points = all_sequences.shape[0]
split_idx = int(split_ratio * n_points)
all_sequences = all_sequences[:, ::, ::]
old_shape = all_sequences.shape[-2:]

data = all_sequences.reshape(all_sequences.shape[0], -1)
training_data = data[:split_idx]
training_gt = ground_truth[:split_idx]
test_data = data[split_idx:]
test_gt = ground_truth[split_idx:]

sfa = SFA(2)

sfa.fit(training_data)
output = sfa.transform(test_data)

gt_delta = np.var(test_gt[1:] - test_gt[:-1], axis=0)
gt_order = np.argsort(gt_delta)
gt_labels = ["x", "y"]

fig, ax = plt.subplots(2, 2, sharex=True)
cutoff = 60
ax[0, 0].plot(output[:cutoff, 0])
ax[1, 0].plot(output[:cutoff, 1])
ax[0, 1].plot(test_gt[:cutoff, gt_order[0]])
ax[1, 1].plot(test_gt[:cutoff, gt_order[1]])
ax[0, 0].set_title("Extracted features")
ax[0, 1].set_title("True position")

plt.tight_layout()
plt.show()

