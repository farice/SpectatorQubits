from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions.unitary import UnitaryGate

from qutip.operators import sigmax
from qutip.qip.operations import snot
from qutip import basis

import numpy as np

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
def update_spectator_analytic_circuit(qc, error_unitary, theta, herm, prep, obs,
                                      parameter_shift):
    inst, qarg, carg = qc.data[1]
    qc.data[1] = UnitaryGate(error_unitary), qarg, carg

    if theta is not None:
        inst, qarg, carg = qc.data[2]
        qc.data[2] = UnitaryGate(
           (obs * (1j * (theta + parameter_shift) * herm / 2).expm() * prep)
        ), qarg, carg

    return qc

# Since we are preparing |+>, it useful to parameterize all unitaries
# considered in this algorithm in terms of their image on this state.
# The classic cos(\theta) |+> + e^(i\phi) |-> representation is used.
def extract_theta_phi(single_qubit_gate):
    # apply gate to |+>
    ket = single_qubit_gate * snot() * basis(2, 0)

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
        phi = np.angle(ket_raw[0].conj() * ket_raw[1] / (
            np.sqrt(ket_raw[0] * ket_raw[0].conj()) * np.sqrt(ket_raw[1] * ket_raw[1].conj())))

    return theta, phi

def get_parameterized_state(theta, phi):
    prepared_basis = [snot() * basis(2, 0), snot() * sigmax() * basis(2, 0)]
    meas = np.cos(theta / 2) * prepared_basis[0] + np.exp(
                1j * phi) * np.sin(theta / 2) * prepared_basis[1]
    return meas.unit()

def get_error_state(unitary):
    return unitary * snot() * basis(2, 0)