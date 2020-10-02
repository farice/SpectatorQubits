from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates.u3 import U3Gate
from qiskit.extensions.unitary import UnitaryGate

import numpy as np


def create_spectator_analytic_circuits(error_unitary, theta, herm, prep, obs):
    # circuit per lo/mid/hi in analytic gradient expression
    qr = [QuantumRegister(1), QuantumRegister(1), QuantumRegister(1)]
    cr = [ClassicalRegister(1), ClassicalRegister(1), ClassicalRegister(1)]
    qc = [QuantumCircuit(qr[0], cr[0]), QuantumCircuit(qr[1], cr[1]), QuantumCircuit(qr[2], cr[2])]

    for _qc, _qr in zip(qc, qr):
        # prepare in X
        _qc.h(_qr)
        # error rotation in euler basis
        _qc.unitary(UnitaryGate(
           error_unitary), _qr)


    qc[0].unitary(UnitaryGate(
       obs * (1j * (theta - np.pi / 4) * herm).expm() * prep
    ), qr[0])

    qc[1].unitary(UnitaryGate(
       obs * (1j * (theta) * herm).expm() * prep
    ), qr[1])

    qc[2].unitary(UnitaryGate(
       obs * (1j * (theta + np.pi / 4) * herm).expm() * prep
    ), qr[2])


    for _qc, _qr, _cr in zip(qc, qr, cr):
        # measure in x-basis
        _qc.h(_qr)
        _qc.measure(_qr, _cr)

    return qc


# explicit update function allows us to avoid creating a new ciruit object
# at every iteration
def update_spectator_analytic_circuits(qc, error_unitary, theta, herm, prep, obs):
    for _qc in qc:
        inst, qarg, carg = _qc.data[1]
        _qc.data[1] = UnitaryGate(error_unitary), qarg, carg

    # update lo/mid/hi in analytic gradient expression
    if theta is not None:
        inst, qarg, carg = qc[0].data[2]
        qc[0].data[2] = UnitaryGate(
           obs * (1j * (theta - np.pi/4) * herm).expm() * prep
        ), qarg, carg

        inst, qarg, carg = qc[1].data[2]
        qc[1].data[2] = UnitaryGate(
           obs * (1j * (theta) * herm).expm() * prep
        ), qarg, carg

        inst, qarg, carg = qc[2].data[2]
        qc[2].data[2] = UnitaryGate(
           obs * (1j * (theta + np.pi/4) * herm).expm() * prep
        ), qarg, carg

    return qc