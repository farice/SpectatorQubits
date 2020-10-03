from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions.unitary import UnitaryGate


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
       obs * (1j * (theta + parameter_shift) * herm / 2).expm() * prep
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
           obs * (1j * (theta + parameter_shift) * herm / 2).expm() * prep
        ), qarg, carg

    return qc