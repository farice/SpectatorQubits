from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions.unitary import UnitaryGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library import IGate

from qutip.operators import sigmax, sigmay, sigmaz
from qutip.qip.operations import snot
from qutip import basis
from qutip import rz

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.ndimage import uniform_filter1d


def create_spectator_analytic_circuit(error_unitary, theta, herm, prep, obs,
                                      parameter_shift):
    # circuit per lo/mid/hi in analytic gradient expression
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)

    # prepare in X
    qc.h(qr)
    # error rotation
    qc.unitary(UnitaryGate(
           error_unitary), qr)
    qc.unitary(UnitaryGate(
       (obs * (1j * (theta + parameter_shift) * herm / 2).expm() * prep)
    ), qr)

    # measure in x-basis
    qc.h(qr)
    qc.measure(qr, cr)

    return qc


# explicit update function allows us to avoid creating a new ciruit object
# at every iteration
def update_spectator_analytic_circuit(qc, error_unitary, theta, herm, prep,
                                      obs, parameter_shift, basis_coin):
    inst, qarg, carg = qc.data[1]
    qc.data[1] = UnitaryGate(error_unitary), qarg, carg

    if theta is not None:
        inst, qarg, carg = qc.data[2]
        qc.data[2] = UnitaryGate(
           (obs * (1j * (theta + parameter_shift) * herm / 2).expm() * prep)
        ), qarg, carg

    if basis_coin % 3 == 0:
        inst, qarg, carg = qc.data[0]
        qc.data[0] = IGate(), qarg, carg
        inst, qarg, carg = qc.data[3]
        qc.data[3] = IGate(), qarg, carg
    elif basis_coin % 3 == 1:
        inst, qarg, carg = qc.data[0]
        qc.data[0] = HGate(), qarg, carg
        inst, qarg, carg = qc.data[3]
        qc.data[3] = HGate(), qarg, carg
    elif basis_coin % 3 == 2:
        hadamard_phase = rz(np.pi / 2) * snot()
        inst, qarg, carg = qc.data[0]
        qc.data[0] = UnitaryGate(hadamard_phase), qarg, carg
        inst, qarg, carg = qc.data[3]
        qc.data[3] = UnitaryGate(hadamard_phase.dag()), qarg, carg

    return qc


# Since we are preparing |+>, it useful to parameterize all unitaries
# considered in this algorithm in terms of their image on this state.
# The classic cos(\theta) |+> + e^(i\phi) |-> representation is used.
def extract_theta_phi(single_qubit_gate, b=snot() * basis(2,0)):
    # apply gate to |+>
    ket = single_qubit_gate * b

    alpha = ket.full()[0][0]
    beta = ket.full()[1][0]
    # rewrite in x-basis
    ket_raw = [(alpha + beta) / 2, (alpha - beta) / 2]
    ket_raw = ket_raw / np.linalg.norm(ket_raw)

    theta = 0
    phi = 0

    if ket_raw[0] * ket_raw[0].conj() < 1e-6:
        theta = np.pi
        phi = 0
    elif ket_raw[1] * ket_raw[1].conj() < 1e-6:
        theta = 0
        phi = 0
    else:
        theta = 2 * np.arccos(np.sqrt(ket_raw[0] * ket_raw[0].conj()))
        phi = (np.angle(ket_raw[0].conj() * ket_raw[1]
               / (np.sqrt(ket_raw[0] * ket_raw[0].conj())
               * np.sqrt(ket_raw[1] * ket_raw[1].conj()))))

    return theta, phi


def get_parameterized_state(theta, phi):
    prepared_basis = [snot() * basis(2, 0), snot() * sigmax() * basis(2, 0)]
    meas = np.cos(theta / 2) * prepared_basis[0] + np.exp(
                1j * phi) * np.sin(theta / 2) * prepared_basis[1]
    return meas.unit()


def get_error_state(unitary):
    return unitary * snot() * basis(2, 0)


def get_error_unitary(sample, sensitivity):
    return (-0.5j * (sample[0] * sigmaz() * sensitivity + sample[1] * sigmay() * sensitivity + sample[2] * sigmax() * sensitivity)).expm()

'''
Plotting utils
'''


def plot_3d_contour(thetas, phis, loss, ax, history):
    history = np.real(history)
    thetas, phis = np.meshgrid(thetas, phis)
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)

    # Interpolate using radial basis kernel.
    # rbf = scipy.interpolate.Rbf(np.tile(thetas, len(phis)), np.repeat(phis, len(thetas)), loss.flatten(),
    #                                 function='linear')
    # loss = rbf(mesh_theta, mesh_phi)
    #  loss = loss.flatten()
    fmax, fmin = loss.max(), loss.min()
    fcolors = (loss - fmin)/(fmax - fmin)
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.viridis(fcolors))
    ax.set_axis_off()

    alphas = np.linspace(0.1, 1, len(history), dtype=np.float32)
    rgba_colors = np.zeros((len(history), 4))
    rgba_colors[:, 2] = 1
    rgba_colors[:, 3] = alphas

    theta = [x[0] for x in history]
    phi = [x[1] for x in history]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    ax.scatter(x, y, z, c=rgba_colors)


def plot_2d_contour(thetas, phis, loss, ax):
    ax.contourf(phis, thetas, loss, alpha=0.4, cmap='viridis')


def plot_2d_contour_scatter(ax, history, color='C0', invert=False):
    history = np.real(history)
    theta = np.array([x[0] for x in history])
    phi = np.array([x[1] for x in history])

    if invert:
        phi = phi + np.pi
    alphas = np.linspace(0.1, 1, len(history) - 1, dtype=np.float32)
    rgba_colors = np.zeros((len(history) - 1, 4))
    rgba_colors[:, 2] = 1
    rgba_colors[:, 3] = alphas

    ax.plot(phi, theta, c=color, alpha=1.0)
#     ax.quiver(phi[:-1], theta[:-1], phi[1:]-phi[:-1], theta[1:]-theta[:-1])
#     ax.scatter(phi, theta, c=color)

    df = pd.DataFrame.from_dict({'phi': phi, 'theta': theta})
    for i, row in df.iterrows():
        if i == 0:
            pass
        else:
            at = ax.annotate('', xy=(row['phi'], row['theta']),
                             xytext=(df.iloc[i-1]['phi'], df.iloc[i-1]['theta']),
                             arrowprops=dict(facecolor=color, width=1,
                                             headwidth=4))
            if len(history) > 0:
                at.set_alpha((i + 1) / len(history))


def plot(frame_idx, elapsed_time, baseline_fidelity=None,
         corrected_fidelity=None,  spectator_fidelity=None,
         context_theta_history=None, correction_theta_history=None,
         context_outcome_hist=None, context_contour=None,
         correction_contour=None, correction_grads=None,
         context_grads=None):
    def set_up_plots():
        _, axs_context_contour = plt.subplots(1, 1, figsize=(7.5, 7.5),
                                              subplot_kw=dict(polar=True))
        _, axs_correction_contour = plt.subplots(1, 2, figsize=(15, 7.5),
                                                 subplot_kw=dict(polar=True))
        _, ax_fid = plt.subplots(1, 1, figsize=(7.5, 5))
        _, ax_hist = plt.subplots(1, 2, figsize=(15, 7.5))
        _, axs_correction_grad = plt.subplots(1, 2, figsize=(15, 10))
        _, ax_context_grad = plt.subplots(1, 1, figsize=(7.5, 5))

        return np.concatenate(([axs_context_contour], axs_correction_contour,
                              [ax_fid], ax_hist, axs_correction_grad,
                               [ax_context_grad]))

    axs = set_up_plots()
    plt.style.use('seaborn')

    if context_contour and context_theta_history:
        plot_2d_contour(
            context_contour['thetas'], context_contour['phis'],
            context_contour['loss'], axs[0])
        plot_2d_contour_scatter(axs[0], context_theta_history)
        axs[0].set_title('Context phase space (gradient steps)')

    if correction_contour and correction_theta_history:
        plot_2d_contour(
            correction_contour[0]['thetas'], correction_contour[0]['phis'],
            correction_contour[0]['loss'], axs[1])
        plot_2d_contour_scatter(axs[1], correction_theta_history[0])
        axs[1].set_title('Context 0: Correction phase space (gradient steps)')
        plot_2d_contour(
            correction_contour[1]['thetas'], correction_contour[1]['phis'],
            correction_contour[1]['loss'], axs[2])
        plot_2d_contour_scatter(axs[2], correction_theta_history[1])
        axs[2].set_title('Context 1: Correction phase space (gradient steps)')

    if baseline_fidelity and corrected_fidelity:
        axs[3].set_title('Fidelity (after burn-in)')
        axs[3].plot(corrected_fidelity, 'g', label='corrected (data)')
        if spectator_fidelity and len(spectator_fidelity) > 0:
            axs[3].plot(spectator_fidelity, 'b', label='corrected (spectator)')
        axs[3].plot(baseline_fidelity, 'r', label='uncorrected')
        axs[3].set_xlabel('Batches')
        axs[3].set_ylabel('Haar-averaged fidelity')
        axs[3].legend()

    if context_outcome_hist:
        max_k = 0
        for k, v in context_outcome_hist.items():
            axs[4].hist(v, label=k)
            max_k = max(k, max_k)
        axs[4].set_title('Distribution of contextual outcomes (all episodes)')

        axs[5].set_title('Distribution of contextual outcomes (most recent episode)')
        axs[5].hist(context_outcome_hist[max_k])

    if correction_grads:
        if 0 in correction_grads.keys():
            axs[6].plot([g[0] for g in correction_grads[0]], label='theta_1')
            axs[6].plot([g[1] for g in correction_grads[0]], label='theta_2')
            axs[6].plot([g[2] for g in correction_grads[0]], label='theta_3')
            axs[6].set_title('Context 0: Correction gradient')
            axs[6].legend()

        if 1 in correction_grads.keys():
            axs[7].plot([g[0] for g in correction_grads[1]], label='theta_1')
            axs[7].plot([g[1] for g in correction_grads[1]], label='theta_2')
            axs[7].plot([g[2] for g in correction_grads[1]], label='theta_3')
            axs[7].set_title('Context 1: Correction gradient')
            axs[7].legend()

    axs[8].plot([g[0] for g in context_grads], label='theta_1')
    axs[8].plot([g[1] for g in context_grads], label='theta_2')
    axs[8].plot([g[2] for g in context_grads], label='theta_3')
    axs[8].set_title('Context gradient')
    axs[8].legend()


def plot_layered(results, context_contour, correction_contour, burnin_length=0,
                 window_size=1):
    def set_up_plots():
        _, axs_context_contour = plt.subplots(1, 1, figsize=(7.5, 7.5),
                                              subplot_kw=dict(polar=True))
        _, axs_correction_contour = plt.subplots(1, 2, figsize=(15, 10),
                                                 subplot_kw=dict(polar=True))
        _, ax_fid = plt.subplots(1, 1, figsize=(7.5, 5))
        _, ax_rel_fid = plt.subplots(1, 1, figsize=(7.5, 5))
        _, ax_rel_centered_fid = plt.subplots(1, 1, figsize=(7.5, 5))

        return np.concatenate(([axs_context_contour], axs_correction_contour,
                              [ax_fid], [ax_rel_fid], [ax_rel_centered_fid]))

    axs = set_up_plots()
    plt.style.use('seaborn')

    plot_2d_contour(
        context_contour['thetas'], context_contour['phis'],
        context_contour['loss'], axs[0])
    axs[0].set_title('Context phase space (gradient steps)')
    plot_2d_contour(
        correction_contour[0]['thetas'], correction_contour[0]['phis'],
        correction_contour[0]['loss'], axs[1])
    axs[1].set_title('Context 0: Correction phase space (gradient steps)')
    plot_2d_contour(
        correction_contour[1]['thetas'], correction_contour[1]['phis'],
        correction_contour[1]['loss'], axs[2])
    axs[2].set_title('Context 1: Correction phase space (gradient steps)')
    for idx, sim in enumerate(results):
        color = f"C{idx % 10}"
        if hasattr(sim, '__len__') and len(sim) > 1:
            sim = sim[0]
        invert = np.real(sim.context_2d_repr[-1][0]) < np.pi / 2
        plot_2d_contour_scatter(axs[0], sim.context_2d_repr, color=color)
        plot_2d_contour_scatter(axs[1], sim.correction_2d_repr[0], color=color,
                                invert=invert)
        plot_2d_contour_scatter(axs[2], sim.correction_2d_repr[1], color=color,
                                invert=invert)

        def fid_plots(data_fids, ctrl_fids, alpha, label, label_override=None):
            label = 'corrected (data)' if idx == 0 and label else ''
            if label_override:
                label = label_override
            axs[3].plot(data_fids, color,
                        label=label,
                        alpha=alpha)
            label = 'uncorrected' if idx == 0 and label else ''
            if label_override:
                label = label_override
            axs[3].plot(ctrl_fids, color,
                        label=label,
                        alpha=alpha, linestyle='--')

            axs[4].plot(data_fids - ctrl_fids, color, alpha=alpha)

            axs[5].plot(data_fids - np.max(ctrl_fids), color, alpha=alpha)

        if hasattr(sim, '__len__') and len(sim) > 1:
            data_fids = np.mean(np.array([s.data_fidelity_per_episode[burnin_length:] for s in sim]), axis=0)
            ctrl_fids = np.mean(np.array([s.control_fidelity_per_episode[burnin_length:] for s in sim]), axis=0)
            fid_plots(data_fids, ctrl_fids, 1.0, label=True)

            for s in sim:
                data_fids = np.array(s.data_fidelity_per_episode[burnin_length:])
                ctrl_fids = np.array(s.control_fidelity_per_episode[burnin_length:])
                fid_plots(data_fids, ctrl_fids, 0.1, label=False)
        else:
            data_fids = uniform_filter1d(np.array(sim.data_fidelity_per_episode[burnin_length:]), window_size)
            ctrl_fids = uniform_filter1d(np.array(sim.control_fidelity_per_episode[burnin_length:]), window_size)
            fid_plots(data_fids, ctrl_fids, 1.0, label=True)

        axs[3].set_title('Fidelity (after burn-in)')
        axs[3].set_xlabel(r'$\Delta t$')
        axs[3].set_ylabel('Haar-averaged fidelity')
        axs[3].legend()

        axs[4].set_xlabel(r'$\Delta t$')
        axs[4].set_ylabel('Haar-averaged fidelity difference')
        axs[4].set_title('Fidelity difference (after burn-in)')

        axs[5].set_xlabel(r'$\Delta t$')
        axs[5].set_ylabel('Haar-averaged fidelity difference')
        axs[5].set_title('Fidelity difference relative to re-centered distribution (after burn-in)')
