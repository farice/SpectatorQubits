import numpy as np

from qutip import *
from qutip.qip.operations import rx
from pylab import *

import functools
import itertools

from numba import njit, jit

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

def keldysh_plot(meas, keldysh_vec, T):
    zs = get_support(T)
    f, a = plt.subplots(1, len(meas), sharex=True, sharey=True, figsize=(8 * len(meas), 6))
    for idx, (k, _) in enumerate(meas.items()):
        corr_vec = keldysh_vec[k]
        for corr in corr_vec:
            a[idx].plot(zs, corr, alpha=0.3)

        a[idx].plot(zs, np.mean(corr_vec, axis=0), 'r+', label='mean')
        title = fr'KQPD ${k}$'
        a[idx].set_title(title)

    f.legend()
    f.show()
    
    
def qobj_to_numpy(keldysh_vec):
    for k in keldysh_vec:
        cv = keldysh_vec[k]
        for i in range(len(cv)):
            for j in range(len(cv[0])):
                cv[i][j] = np.real((np.array(cv[i][j])))
        keldysh_vec[k] = np.squeeze(np.array(cv))
    return keldysh_vec