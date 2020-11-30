import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from qutip import *

import itertools


# Classical stochastic parameterization of sigmax() rotation.
def gaussian_process(prev, i, mu, sigma, T):
    if i > T:
        return prev
    i += 1
    prev.append(np.random.normal(mu, sigma) * sigmax())
    return gaussian_process(prev, i, mu, sigma, T)


# Cached.
def hamiltonian(gp, t, *args):
    # t is rounded since we're working on the lattice.
    t = int(t)
    return gp[t]


def get_support(T):
    def mul(x):
        return x[0] * x[1]
    zs = list(itertools.product(range(T + 1), [1, -1]))
    zs = list(set(map(mul, zs)))
    return np.sort(zs)


def keldysh_vec_to_pd(keldysh_vec, T):
    d = {
        'basis': np.array([]),
        'p(z)': np.array([]),
        'z': np.array([]),
        'random_gp': np.array([]),
        }
    zs = get_support(T)
    for k, v in keldysh_vec.items():
        for idx, random_gp_v in enumerate(v):
            d['basis'] = np.append(d['basis'], np.repeat(k, len(random_gp_v)))
            d['p(z)'] = np.append(d['p(z)'], random_gp_v)
            d['z'] = np.append(d['z'], zs)
            d['random_gp'] = np.append(d['random_gp'], np.repeat(idx, len(random_gp_v)))
    return pd.DataFrame(d)


def keldysh_plot(meas, keldysh_vec, T):
    plt.style.use('seaborn')
    zs = get_support(T)
    f, a = plt.subplots(int(len(meas) / 3), 3, sharex=True, sharey=True,
                        figsize=(24, 6 * len(meas) / 3))
    # f.suptitle('Keldysh quasi-probability distributions for single-qubit fixed-axis GP error')
    for idx, (k, _) in enumerate(meas.items()):
        ax = a[idx // 3, idx % 3]
        corr_vec = keldysh_vec[k]
        for corr in corr_vec:
            ax.plot(zs, corr, alpha=0.3)

        ax.plot(zs, np.mean(corr_vec, axis=0), 'r', label='mean')
        title = fr'${k}$'
        ax.set_title(title)
        ax.set_ylabel(r'$qp(z)$')
        ax.set_xlabel(r'$z$')
        ax.legend()
    f.show()


def keldysh_bdry_plot(meas, keldysh_bdry_vec, T, eig=None):
    plt.style.use('seaborn')
    zs = get_support(T)
    f, a = plt.subplots(int(len(meas) / 3), 3, sharex=True, sharey=True,
                        figsize=(24, 6 * len(meas) / 3))
    # f.suptitle('Keldysh quasi-probability distributions for single-qubit fixed-axis GP error')
    for idx, (k, _) in enumerate(meas.items()):
        ax = a[idx // 3, idx % 3]
        corr_vec = keldysh_bdry_vec[k]

        labels = {0 : r"$-1$ at boundary $\tau$", 1 : r"$+1$ at boundary $\tau$"}
        for corr in corr_vec:
#             print(corr[:, 0])
            if eig == 1:
                ax.plot(zs, corr[:, 1], alpha=0.3, label=labels[1])
                labels[1] = "_nolegend_"
            elif eig == -1:
                ax.plot(zs, corr[:, 0], alpha=0.3, label=labels[0])
                labels[0] = "_nolegend_"
            else:
                ax.plot(zs, corr[:, 1], alpha=0.3, label=labels[1])
                labels[1] = "_nolegend_"
                ax.plot(zs, corr[:, 0], alpha=0.3, label=labels[0])
                labels[0] = "_nolegend_"

        if eig == 1:
            ax.plot(zs, np.mean(corr_vec[:, :, 1], axis=0), 'r', label='mean (+1)')
        elif eig == -1:
            ax.plot(zs, np.mean(corr_vec[:, :, 0], axis=0), 'b', label='mean (-1)')
        else:
            ax.plot(zs, np.mean(corr_vec[:, :, 0], axis=0), 'b', label='mean (-1)')
            ax.plot(zs, np.mean(corr_vec[:, :, 1], axis=0), 'r', label='mean (+1)')
        title = fr'${k}$'
        ax.set_title(title)
        ax.set_ylabel(r'$qp(z)$')
        ax.set_xlabel(r'$z$')
        ax.legend()
    f.show()


def qobj_to_numpy(keldysh_vec):
    for k in keldysh_vec:
        cv = keldysh_vec[k]
        for i in range(len(cv)):
            for j in range(len(cv[0])):
                cv[i][j] = np.real((np.array(cv[i][j])))
        keldysh_vec[k] = np.squeeze(np.array(cv))
    return keldysh_vec