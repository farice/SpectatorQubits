# %% codecell
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, BasicAer, execute
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import random
import numba
from qutip import *

%matplotlib inline

# %% markdown
# Things I need to look at:
# 1. Assume a function f(x) that generates the gaussian distributions (relatively well behaved). How does a strategy like this works then?
# My assumption is that the more stochastic the errors are in the exploration phase, the better MAB will learn. If MAB sees more data that are not similar then it'll train better.
#
# 2. Policy gradient
# Assume the action space is not discrete but use gradient descent to find out the best action.
#
# 3. Put error bars on that average plot.
# 4. Time varying error. How do we balance between exploraion and exploitation.
# Use hypothesis testing. Null hypothesis is that the error does not change, alternative hypothesis is that it changes. Use how fast v0_mean_estimate or v1_mean_estimate changes to calculate some p value that will tell us that the error is changing.
# 5. Optimize MAB parameters.
# 6. How to extend this to more complicated error types?
# 7. Find composite pulses resistant against over-rotation and phase error.


# %% codecell
# prepare state in X basis
# rotate along Z axis by theta
# measure in Y basis
# hence, p(+1) = cos^2[(pi/2-theta)/2] allowing us to distinguish between
# small positive and negative errors
# note, we cannot distinguish between rotations by (pi/2 + eps, pi/2 - eps)
# therefore, we assume errors lie within [-pi/2, pi/2]
def get_spectator_context_circuit(error_theta):
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)

    qc.h(qr)

    qc.rz(error_theta, qr)

    qc.sdg(qr)
    qc.h(qr)
    qc.measure(qr, cr)

    return qc


# %% codecell
# prepare state in X basis
# rotate along Z axis by theta
# measure in Y basis
# hence, p(+1) = cos^2[theta/2] allowing us to evaluation error correction reward
def get_spectator_reward_circuit(error_theta):
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)

    qc.h(qr)

    qc.rz(error_theta, qr)

    qc.h(qr)
    qc.measure(qr, cr)

    return qc


# %% codecell
qc = get_spectator_reward_circuit(0.1 * np.pi)
qc.draw()


# %% codecell
qc = get_spectator_context_circuit(0.1 * np.pi)
qc.draw()

# %% codecell
qc_small_pos = get_spectator_context_circuit(0.25 * np.pi)
qc_small_neg = get_spectator_context_circuit(-0.25 * np.pi)
sim_pos = execute(
    qc_small_pos, backend=BasicAer.get_backend('qasm_simulator'),
    shots=1000)
sim_neg = execute(
    qc_small_neg, backend=BasicAer.get_backend('qasm_simulator'),
    shots=1000)
print(sim_pos.result().get_counts())
print(sim_neg.result().get_counts())

# %% codecell
# spec = [
#     ('thetas', double[:]),               # a simple scalar field
#     ('rewards', double[:]),          # an array field
# ]


# (action, reward) distribution for a given state
# in particular, we have two states: V0, V1
# @jitclass(spec)
class MDPNode:
    def __init__(self, num_arms):
        # action set
        self.thetas = np.pi/2 * np.linspace(-1, 1, num_arms)
        # correspondingly indexed reward set
        self.rewards = np.zeros(num_arms)

    def optimal_theta(self):
        return self.thetas[np.argmax(self.rewards)]


# %% codecell
def mab(error_samples, num_arms=11, N=10000):
    outcomes = np.zeros(N)
    V0 = MDPNode(num_arms)
    V1 = MDPNode(num_arms)

    process_fidelity_corrected = np.zeros(N)
    process_fidelity_noop = np.zeros(N)
    for i in range(N):
        # print(V0.rewards)
        # print(V1.rewards)
        # print("\n")

        spectator_qc_1 = get_spectator_context_circuit(error_samples[i])

        # single measurement of first spectator qubit
        sim_1 = execute(
            spectator_qc_1, backend=BasicAer.get_backend('qasm_simulator'),
            shots=1)
        outcome_1 = int(
            list(sim_1.result().get_counts().keys())[0]
            )
        outcomes[i] = outcome_1

        # print(outcome_1)
        # print("\n")

        # contextual multi-arm bandit
        (rewards, thetas) = (V0.rewards, V0.thetas) if outcome_1 == 0 else (V1.rewards, V1.thetas)

        eps = np.random.rand()
        arm = np.random.randint(0, num_arms) if eps <= 0.1 or i < N/5 else np.argmax(
                                                                      rewards)
        correction_theta = thetas[arm]

        # rotations along the same axis commute
        spectator_qc_2 = get_spectator_reward_circuit(
            error_samples[i] + correction_theta)
        sim_2 = execute(
            spectator_qc_2, backend=BasicAer.get_backend('qasm_simulator'),
            shots=1)
        outcome_2 = int(
            list(sim_2.result().get_counts().keys())[0]
            )

        if (outcome_2 == 0):
            rewards[arm] += 1
        else:
            rewards[arm] -= 1

        process_fidelity_corrected[i] = rz(error_samples[i] + correction_theta).tr() / 2
        process_fidelity_noop[i] = rz(error_samples[i]).tr() / 2

        # rewards[arm] = process_fidelity_corrected[i]

        # print(rz(error[i] + correction_theta))
        # print(process_fidelity_corrected[i], process_fidelity_noop[i])
        # print("\n")

    return (V0, V1,
            process_fidelity_corrected, process_fidelity_noop, outcomes)


# %% codecell
# mean of gaussian error distribution
mu_list = np.pi * np.array([0.0])
alpha = np.pi * 0.5

theta_sequence = []
outcomes_sequence = []
fid_corrected_sequence = []
fid_noop_sequence = []

N = 1000
for mu in mu_list:
    error_samples = np.random.uniform(-alpha, alpha, N)
    V0, V1, fid_corrected, fid_noop, outcomes = mab(error_samples, N=N)

    theta_sequence.append((V0.optimal_theta(), V1.optimal_theta()))
    outcomes_sequence.append(outcomes)
    fid_corrected_sequence.append(fid_corrected)
    fid_noop_sequence.append(fid_noop)


# %% codecell
(theta_sequence)
# %% codecell
(outcomes_sequence)
# %% codecell
(fid_corrected_sequence)
# %% codecell
(fid_noop_sequence)

# %% markdown
In this case, we have a positive biased error distribution drawn from a Gaussian(mu=0.5, sigma=0.1)
We expect the error discrimination procedure:
- prepare: X +1 eigenstate
- error: rotate along Z axis
- measure: X basis

to work quite well given that we essentially are using multi-armed bandits to search for the appropriate cancelling rotation.

# %% codecell
idx = np.linspace(1, N, N)
plt.figure()
plt.plot(idx, fid_noop_sequence[0], 'r.', idx, fid_corrected_sequence[0], 'g.')
plt.show()
