from __future__ import division
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence
import numpy as np
import math

class SparsificationProblem:

    def __init__(self,edges):
        self.m = len(edges)
        self.edges = (np.array([e[0] for e in edges], dtype=int), np.array([e[1] for e in edges], dtype=int))
        self.n = max(max(self.edges[0]), max(self.edges[1])) + 1
        self.weights = np.array([e[2] for e in edges], dtype=float)
        self.last_y = np.random.random(self.n)
        self.norm = self.value(np.zeros(self.m))

    def __call__(self, x):
        return 1 - self.value(x)/self.norm

    def delta_laplacian(self,x):
        x = np.asarray(x)
        w = self.weights * (1 - x)
        w = np.concatenate((w, w))
        e = (
             np.concatenate((self.edges[0], self.edges[1])),
             np.concatenate((self.edges[1], self.edges[0]))
        )
        A = sp.csr_matrix((w,e), shape=(self.n, self.n))
        D = sp.dia_matrix((A.sum(axis=1).transpose(), 0), shape=(self.n, self.n))
        return D-A

    def leading_eigenvalue(self, x, n=1):
        L = self.delta_laplacian(x)
        v0 = self.last_y + 1E-6*np.random.randn(self.n)
        try:
            l, y = eigsh(L, k=n, which='LM', v0=v0, tol=1E-6, ncv=40)
        except ArpackNoConvergence as err:
            print(err)
            l = err.eigenvalues
            y = err.eigenvectors
        index = [i for i, v in sorted(enumerate(l), key=lambda x: x[1], reverse=True)]
        l = np.real(l[index])
        y = np.real(y[:, index])
        for i in range(n):
            y[:, i] /= norm2(y[:, i])
        return l, y

    def value(self, x):
        l, y = self.leading_eigenvalue(x)
        self.last_y = y[:, 0]
        return l[0]

    def gradient(self, x):
        l, y = self.leading_eigenvalue(x, 2)
        if (l[0] - l[1]) / l[0] < 1E-6:
            y = (y[:, 0] + y[:,1]) / 2
        else:
            y = y[:,0]
        return -self.weights * (y[self.edges[0]] - y[self.edges[1]])**2

    def last_gradient(self):
        return -self.weights * (self.last_y[self.edges[0]] - self.last_y[self.edges[1]])**2

    def reset(self):
        self.last_y = np.random.random(self.n)
        self.x = np.zeros(self.m)

    def sample_edges(self, x):
        edges = [(self.edges[0][i], self.edges[1][i], self.weights[i]) for i in range(self.m) if x[i] > np.random.random()]
        return edges


def norm(x):
    return max(abs(x))


def norm2(x):
    return np.sqrt(np.dot(x,x))


def linear_sparsification_profile(edges, x_values, by_weight=False, key=lambda edge: edge[2]):
    """
    Return MASS profile for sparsifying edges

    Parameters
    ----------
    edges : iterable of tuples
    x_values : iterable, fraction of edges or weight to remove
    by_weight : bool
        switch to remove percent of total weight
    key : key function for sorting edges

    Returns
    -------
    iterable

    """
    edges.sort(key=key)
    if by_weight:
        select_key = np.asarray(accumulate((e[2] for e in edges)))
        select_key /= select_key[-1]
    else:
        select_key = np.array([k/len(edges) for k in range(1,len(edges)+1)])

    sp = SparsificationProblem(edges)
    val = []
    for xv in x_values:
        x = xv < select_key
        val.append(sp(x))
    return val


import networkx as nx
import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('LFR.mat')
G=nx.from_numpy_matrix(mat['A'])
list(G.edges(data=True))[0:3]
edges = list(G.edges.data('weight'))
list = range(21)[::1]
thresholds = [x / 20 for x in list]
plt.plot(linear_sparsification_profile(edges, thresholds))
plt.show()
