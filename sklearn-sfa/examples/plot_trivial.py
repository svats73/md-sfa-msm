"""
=============================
Fill Modes (for Missing Dimensions)
=============================

An example plot of the different fill modes of :class:`sksfa.SFA`.

The input data does only have effective dimension 2 (the other dimensions are linearly dependent),
but 3 slow features are set to be extracted. The different fill_mode options will fill the missing
features in different ways.
Setting 'fill_mode' to None, would make the transformer throw an exception.
"""


import numpy as np
from sksfa import SFA
import matplotlib.pyplot as plt


n_samples = 500
n_slow_features = 3
dim = 4
fill_modes = ["noise", "zero", "fastest"]

# Generate different randomly shifted time-scales
t = np.linspace(0, 2*np.pi, n_samples).reshape(n_samples, 1)

# Generate latent cosine signals
x = np.hstack([np.cos(t), 0.5 * np.cos(t), np.cos(2 * t), 1.5 * np.cos(t)])

# Compute random affine mapping of cosines (observed)
A = np.random.normal(0, 1, (dim, dim))
b = np.random.normal(0, 2, (1, dim))
data = np.dot(x, A) + b

# Extract slow features from observed data

# Plot cosines, mapped data, and extracted features
fig, ax = plt.subplots(2 + len(fill_modes), 1, sharex=True)
fig.set_size_inches(8, 18)
fig.subplots_adjust(hspace=0.5)
for d in range(n_slow_features):
    ax[0].plot(x[:, d])
ax[1].plot(data)
for idx, fill_mode in enumerate(fill_modes):
    sfa = SFA(n_slow_features, fill_mode=fill_mode)
    slow_features = sfa.fit_transform(data)
    ax[2 + idx].plot(slow_features[:, :-1])
    ax[2 + idx].plot(slow_features[:, -1], linestyle=":", color="purple")
    ax[2 + idx].set_title(f"Extracted features, fill_mode='{fill_mode}'")
    ax[2 + idx].set_xlabel("Time t")
ax[0].set_title("x(t)")
ax[1].set_title("A⋅x(t) + b")
for idx in range(2 + len(fill_modes)):
    ax[idx].set_ylabel("Features")
plt.show()
