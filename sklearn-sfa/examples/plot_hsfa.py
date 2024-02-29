"""
============
Moving Digit (Experimental - Hierarchical SFA)
============
An example of :class:`sksfa.HSFA` applied to a simple image time-series:
a one-digit version of the moving MNIST dataset. Each data point is 4096-dimensional.

.. image:: ../images/moving_mnist.gif
   :align: center

If the change in x is not significantly faster or slower than the change in y, HSFA with only two output features
successfully extracts a smooth (and possibly flipped) representation of the position of the digit in the image.

Ground truth is only added for the comparison, not during training.

(Note that a problem like this can also be solved with linear SFA. This example serves the purpose of providing an
example on how an HSFA network is initialized and how it can be directly applied to image data without flattening.)
"""

import numpy as np
from sksfa import HSFA
import matplotlib.pyplot as plt
import os

# Loading and preparing the data
# - HSFA requires a colorchannel, even for grayscale images
split_ratio = 0.7
data = np.load("data/mmnist_data.npy").squeeze()[..., None]
ground_truth = np.load("data/mmnist_positions.npy").squeeze()
n_points = data.shape[0]
split_idx = int(split_ratio * n_points)

training_data = data[:split_idx]
training_gt = ground_truth[:split_idx]
test_data = data[split_idx:]
test_gt = ground_truth[split_idx:]

# Preparing the HSFA-network:
# - each layer needs a 6-tuple for configuration
# - each 6-tuple contains (kernel_width, kernel_height, stride_width, stride_height, n_features, expansion_degree)
# The final layer will always be a full connected SFA layer
layer_configurations = [(8, 8, 8, 8, 8, 1),
                        (2, 2, 2, 2, 8, 2)]

hsfa = HSFA(n_components=2,
            input_shape=data.shape[1:],
            layer_configurations=layer_configurations,
            internal_batch_size=100,
            noise_std=0.01)

hsfa.summary()

hsfa.fit(training_data)
output = hsfa.transform(test_data)

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
