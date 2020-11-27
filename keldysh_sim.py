import numpy as np

from qutip import *
from qutip.qip.operations import rx
from pylab import *

import functools
import itertools

from numba import njit, jit

import keldysh_utils as utils

def generate_contour_trajectory_sums(m, T):
    r = []

    sgn = -1 if m < 0 else 1
    for m_L in range(0, sgn * (T + 1), sgn):
        m_R = 2*m - m_L
        if np.abs(m_R) <= T:
            r.append((m_L, m_R))
    return r


def generate_contour_trajectories_for_sums(s, T):
    def integer_division(a, b):
        return -(-a // b) if a < 0 else a // b
    ones =  T // 2 + integer_division(s, 2)
    zeros = T // 2 - integer_division(s, 2)
    if np.abs(s) % 2 == 1 and s < 0:
        zeros += 1
    elif np.abs(s) % 2 == 1 and s > 0:
        ones += 1

    path = np.append(np.ones(ones, dtype=np.int32),
                     np.zeros(zeros, dtype=np.int32))

    return np.array(list(set(itertools.permutations(path, len(path)))))


@njit # Ridiculous improvement in runtime. Optimizes linalg and loops.
def keldysh_qprob(observable_spectra, q_prob, H, l_perms, r_perms, state0, T):
    for l_perm in l_perms:
        for r_perm in r_perms:
            # Delta on L and R path end-points.
            if l_perm[-1] != r_perm[-1]:
                continue
            base_prob = (observable_spectra[l_perm[0]]).conjugate().transpose().dot(
                state0).dot(
                (state0).conjugate().transpose()).dot(
                observable_spectra[r_perm[0]])
            A_L = np.array([[1+0j]])
            A_R = np.array([[1+0j]])
            for i in range(T - 1):
                eig_l_ket = observable_spectra[l_perm[i]]
                eig_l_bra = (observable_spectra[l_perm[i+1]]).conjugate().transpose()
                A_L *= eig_l_bra.dot(H[i]).dot(eig_l_ket)

                eig_r_ket = observable_spectra[r_perm[i]]
                eig_r_bra = (observable_spectra[r_perm[i+1]]).conjugate().transpose()
                A_R *= eig_r_bra.dot(H[i]).dot(eig_r_ket)

            q = A_L * A_R.conjugate() * base_prob
            # assert np.imag(q) == 0
            q_prob += q
    return q_prob


def sample_gps(meas_names, meas_eigs, M, mu, sigma, T, state0):
    zs = utils.get_support(T)
    keldysh_vec = {}

    for _ in range(M):
        # Sample GP.
        gp = utils.gaussian_process([], 0, mu, sigma, T)
        h = np.array([np.array((-1j * utils.hamiltonian(gp, i)).expm()) for i in range(T - 1)])
        for n, e in zip(meas_names, meas_eigs):
            qd_i = []
            for z in zs:
                sums = generate_contour_trajectory_sums(z, T)
                q_prob = np.array([[0+0j]])
                for s_L, s_R in sums:
                    if (T % 2 == 1 and (s_L % 2 == 0 or s_R % 2 == 0)) \
                      or (T % 2 == 0 and (s_L % 2 == 1 or s_R % 2 == 1)):
                        continue
                    l_perms = generate_contour_trajectories_for_sums(s_L, T)
                    r_perms = generate_contour_trajectories_for_sums(s_R, T)
                    q_prob = keldysh_qprob(e, q_prob, h, l_perms, r_perms, state0, T)
                qd_i.append(q_prob)
            if n in keldysh_vec:
                keldysh_vec[n].append(qd_i)
            else:
                keldysh_vec[n] = [qd_i]
    return keldysh_vec


