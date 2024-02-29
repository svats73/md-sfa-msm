"""
======================================
SFA to Approximate Spectral Embeddings
======================================

Slow Feature Analysis can be interpreted as a function approximation version
of a spectral embedding, specifically Laplacian Eigenmaps (LEMs).

At their core, LEMs eigen-decompose a graph-derived matrix and use the eigenvectors
as embedding. The graph is either given or generated from data, e.g., by connecting
nearest neighbors.

To use standard SFA, a time-series is generated using this graph by performing a
random walk on it. The actual time-series is the sequence of features of visited nodes,
their embedding is the mapping learned by SFA.

Benefits:

* unseen points can be embedded without Nyström approximation.
* instead of calculating the eigen-decomposition on the N-by-N Laplacian, only d-by-d covariance matrices need to be decomposed.
* specific choice of the model family used for embedding.

Downsides:

* depends on randomly sub-sampled data.
* found mapping might fail to extract structure due to limited model class (linear/extended linear).

.. admonition:: Note

   The described method works, but is not necessarily efficient. The not-yet-implemented
   Graph-based version of SFA (GSFA) is generally the smarter choice and does not require
   a random walk.

Below you see an example of random-walk SFA applied to a "wavy circle" dataset (N=1000) that clearly exhibits structure.
A dimensionality reduction using PCA does not work, because the data has almost
isotropic covariance. An embedding using LEMs leaves the global structure intact, but cannot naturally be applied to unseen points.
SFA finds a linear mapping to the same end, while also generalizing embedding of unseen points.

LEMs and the random-walk use a 50-nearest-neighbor embedding. The random-walk time-series consists of three sequences with
500 steps each.
"""
import numpy as np
import scipy as sp
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding as LEM
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sksfa import SFA
from sksfa.utils import randomWalkFromCSC

k = 0.125
n_points = 1000

u = np.linspace(0, 2 * np.pi / k, n_points)
phase_offset = 0.25 * np.pi
v =  k * u + phase_offset


data = np.zeros((n_points, 3))
if True:
    data[:, 0] = np.cos(v)
    data[:, 1] = np.sin(v)
    data[:, 2] = np.cos(u)
else:
    a, b = 10, 5
    data[:, 0] = (a + b * np.cos(u))*np.cos(v)
    data[:, 1] = (a + b * np.cos(u))*np.sin(v)
    data[:, 2] = b * np.sin(u)

lem = LEM(2, n_neighbors=50)
embedded = lem.fit_transform(data)
A = lem.affinity_matrix_

restart_rate = 500
n_random_samples = 1500
trajectory = randomWalkFromCSC(sp.sparse.csc_matrix(A), n_random_samples, restart_rate=restart_rate)
walk_data = data[trajectory]

visited = np.unique(trajectory)
non_visited = np.setdiff1d(np.arange(0, n_points), visited)

pf = PolynomialFeatures(1)
sfa = SFA(2, batch_size=restart_rate if restart_rate > 0 else None)
sf = sfa.fit_transform(pf.fit_transform(walk_data))
oos = sfa.transform(pf.transform(data[non_visited]))

pca = PCA(2)
pc = pca.fit_transform(data)

fig = plt.figure()
fig.set_size_inches(8, 12)
fig.subplots_adjust(hspace=0.5)

ax_3d = fig.add_subplot(321, projection='3d')
ax_3d.set_title("Wavy circle data")
ax_rw = fig.add_subplot(323, projection='3d')
ax_rw.set_title("Random walk samples")
ax_lem = fig.add_subplot(322)
ax_lem.set_title("Spectral embedding")
ax_sfa= fig.add_subplot(324)
ax_sfa.set_title("Linear SFA (in-sample)")
ax_oos= fig.add_subplot(325)
ax_oos.set_title("Linear SFA (out-of-sample)")
ax_pca= fig.add_subplot(326)
ax_pca.set_title("PCA embedding")

ax_3d.scatter(data[:, 0], data[:, 1], data[:, 2], c=u, cmap="hsv", s=2)
ax_rw.scatter(walk_data[:, 0], walk_data[:, 1], walk_data[:, 2], c=u[trajectory], cmap="hsv", s=2)

ax_lem.scatter(embedded[:, 0], embedded[:, 1], c=u, cmap="hsv", s=3)
ax_sfa.scatter(sf[:, 0], sf[:, 1], c=u[trajectory], cmap="hsv", s=3)
ax_oos.scatter(oos[:, 0], oos[:, 1], c=u[non_visited], cmap="hsv", s=3)
ax_pca.scatter(pc[:, 0], pc[:, 1], c=u, cmap="hsv", s=3)
plt.show()
