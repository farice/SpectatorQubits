from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions.standard.rz import RZGate
from qiskit.extensions.standard.ry import RYGate
from qiskit.circuit.library.standard_gates.u3 import U3Gate
from qiskit.extensions.unitary import UnitaryGate
from qutip.operators import sigmax, sigmay, sigmaz
from qutip.qip.gates import rotation


def create_spectator_context_circuit(error_theta, error_phi, error_lambda, measure_theta, measure_phi, measure_lambda):
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)

    # prepare in X
    qc.h(qr)

    # error rotation in euler basis
    qc.u3(error_theta, error_phi, error_lambda, qr)

    # measure in arbitrary basis
    qc.u3(measure_theta, measure_phi, measure_lambda, qr)
    # qc.unitary(UnitaryGate(
    #    rotation(
    #       measure_bloch_vector[0] * sigmax() + measure_bloch_vector[1] * sigmay() + measure_bloch_vector[2] * sigmaz(),
    #        measure_theta)
    #    ), qr)
    qc.measure(qr, cr)

    return qc


# reward is simply fidelity between prepared pure state and post-error-and-correction state
# errors are corrections are both unitary
def create_spectator_reward_circuit(error_theta, error_phi, error_lambda, correction_theta, correction_bloch_vector):
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)

    # prepare in X
    qc.h(qr)

    # error rotation in euler basis
    qc.u3(error_theta, error_phi, error_lambda, qr)

    # correction rotation as (theta, Bloch vector)
    qc.unitary(UnitaryGate(
        rotation(
            correction_bloch_vector[0] * sigmax() + correction_bloch_vector[1] * sigmay() + correction_bloch_vector[2] * sigmaz(),
            correction_theta)
        ), qr)

    # measure in x-basis
    qc.h(qr)
    qc.measure(qr, cr)

    return qc


# explicit update function allows us to avoid creating a new ciruit object
# at every iteration
def update_spectator_circuit(qc, error_theta, error_phi, error_lambda, correction_theta=None, correction_bloch_vector=None):
    inst, qarg, carg = qc.data[1]

    qc.data[1] = U3Gate(error_theta, error_phi, error_lambda), qarg, carg

    if correction_theta is not None:
        qc.data[2] = UnitaryGate(
        rotation(
            correction_bloch_vector[0] * sigmax() + correction_bloch_vector[1] * sigmay() + correction_bloch_vector[2] * sigmaz(),
            correction_theta)
        ), qarg, carg
