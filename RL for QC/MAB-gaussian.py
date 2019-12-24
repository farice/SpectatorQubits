# %% codecell
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, BasicAer, execute
from qiskit.extensions.standard.rz import RZGate
import matplotlib.pyplot as plt
import numpy as np
from qutip import *


# %% markdown
For the state - determining circuit(where we mean state in the MDP and not quantum sense), we:
- prepare in X(+1) eigenstate
- rotate along Z axis by error theta
- measure in Y basis
Hence, p(+i) = cos ^ 2[(pi / 2 - theta) / 2] allowing us to distinguish between small positive and negative errors.
Note, we cannot distinguish between rotations by(pi / 2 + eps, pi / 2 - eps). Therefore, we assume errors lie within[-pi / 2, pi / 2].


# %% codecell
def create_spectator_context_circuit(error_theta):
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)

    qc.h(qr)

    qc.rz(error_theta, qr)

    qc.sdg(qr)
    qc.h(qr)
    qc.measure(qr, cr)

    return qc


# explicit update function allows us to avoid creating a new ciruit object
# at every iteration
def update_spectator_circuit(qc, error_theta):
    inst, qarg, carg = qc.data[1]
    qc.data[1] = RZGate(error_theta), qarg, carg


# %% markdown
For the reward - determining circuit, we:
- prepare in X(+1) eigenstate
- rotate along Z axis by error theta
- measure in X basis
Hence, p(+1) = cos ^ 2[theta / 2] allowing us to evaluation error correction reward.


# %% codecell
def create_spectator_reward_circuit(error_theta):
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)

    qc.h(qr)

    qc.rz(error_theta, qr)

    qc.h(qr)
    qc.measure(qr, cr)

    return qc


# %% codecell
qc = create_spectator_context_circuit(0.1 * np.pi)
qc.draw()

# %% codecell
qc_small_pos = create_spectator_context_circuit(0.1 * np.pi)
qc_small_neg = create_spectator_context_circuit(-0.1 * np.pi)
sim_pos = execute(
    qc_small_pos, backend=BasicAer.get_backend('qasm_simulator'),
    shots=1000)
sim_neg = execute(
    qc_small_neg, backend=BasicAer.get_backend('qasm_simulator'),
    shots=1000)
print(sim_pos.result().get_counts())
print(sim_neg.result().get_counts())


# %% codecell
# (action, reward) distribution for a given state
# in particular, we have two states: V0, V1
# @jitclass(spec)
class MDPNode:
    def __init__(self, num_arms):
        # action set
        self.thetas = np.pi / 2 * np.linspace(-1, 1, num_arms)
        # correspondingly indexed (reward | state, action) set
        # samples from beta(S, F) distribution
        self.estimated_rewards = np.ones(num_arms, dtype=np.float64)
        # (successes | arm)
        self.S = np.ones(num_arms, dtype=np.int)
        # (failures | arm)
        self.F = np.ones(num_arms, dtype=np.int)

    def resample_rewards(self):
        # estimated_rewards is drawn from beta(S, F) distribution for each arm
        # which enables exploration
        for i in range(len(self.estimated_rewards)):
            self.estimated_rewards[i] = np.random.beta(self.S[i], self.F[i])

    def optimal_theta(self):
        return self.thetas[np.argmax(self.estimated_rewards)]

    def success(self):
        # update assuming success occurred when pulling currently optimal arm
        arm = np.argmax(self.estimated_rewards)
        self.S[arm] += 1

    def failure(self):
        arm = np.argmax(self.estimated_rewards)
        self.F[arm] += 1


# %% codecell
def mab(error_samples, num_arms=11):
    N = len(error_samples)
    outcomes = np.zeros(N)
    V0 = MDPNode(num_arms)
    V1 = MDPNode(num_arms)

    process_fidelity_corrected = np.zeros(N)
    process_fidelity_noop = np.zeros(N)

    spectator_context_qc = create_spectator_context_circuit(0)
    spectator_reward_qc = create_spectator_reward_circuit(0)
    for i in range(N):
        update_spectator_circuit(spectator_context_qc, error_samples[i])

        # single measurement of first spectator qubit
        sim_1 = execute(
            spectator_context_qc, backend=BasicAer.get_backend(
                'qasm_simulator'),
            shots=1)
        outcome_1 = int(
            list(sim_1.result().get_counts().keys())[0]
        )
        outcomes[i] = outcome_1

        # contextual multi-arm bandit
        context = V0 if outcome_1 == 0 else V1
        context.resample_rewards()
        correction_theta = context.optimal_theta()

        # rotations along the same axis commute
        update_spectator_circuit(spectator_reward_qc,
                                 error_samples[i] + correction_theta)
        sim_2 = execute(
            spectator_reward_qc,
            backend=BasicAer.get_backend('qasm_simulator'), shots=1)
        outcome_2 = int(
            list(sim_2.result().get_counts().keys())[0]
        )

        if (outcome_2 == 0):
            context.success()
        else:
            context.failure()

        process_fidelity_corrected[i] = rz(
            error_samples[i] + correction_theta).tr() / 2
        process_fidelity_noop[i] = rz(error_samples[i]).tr() / 2

    return (V0, V1,
            process_fidelity_corrected, process_fidelity_noop, outcomes)


# %% codecell
# unif [-alpha, alpha]
alpha_list = np.pi * np.array([0.5])

V0_sequence, V1_sequence = [], []
outcomes_sequence = []
fid_corrected_sequence, fid_noop_sequence = [], []

N = 1000
for alpha in alpha_list:
    error_samples = np.random.uniform(-alpha, alpha, N)
    V0, V1, fid_corrected, fid_noop, outcomes = mab(error_samples)

    V0_sequence.append(V0)
    V1_sequence.append(V1)
    outcomes_sequence.append(outcomes)
    fid_corrected_sequence.append(fid_corrected)
    fid_noop_sequence.append(fid_noop)


# %% codecell
print(V0_sequence[0].estimated_rewards)
print(V1_sequence[0].estimated_rewards)


# %% codecell
idx = np.linspace(1, N, N)
plt.figure()
plt.plot(idx, fid_noop_sequence[0], 'r.', idx, fid_corrected_sequence[0], 'g.')
plt.show()
