import numpy as np
from typing import Optional

from gym import Env, Space
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, BasicAer, execute
from qiskit.extensions.standard.rz import RZGate
import matplotlib.pyplot as plt
import numpy as np
from qutip import *


class SpectatorEnv(Env):
    def __init__(self,
                 error_samples,
                 num_arms: int = 21):

        self.spectator_context_qc = self.create_spectator_context_circuit(0)
        self.spectator_reward_qc = self.create_spectator_reward_circuit(0)

        self.num_arms = num_arms
        self.error_samples = error_samples

        self.action_space = Discrete(num_arms)
        self.observation_space = MultiBinary(1)
        self.ep_length = len(error_samples)
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self):
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state

    def step(self, action):
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, self.info

    def _choose_next_state(self):
        self.update_spectator_circuit(self.spectator_context_qc, self.error_samples[self.current_step])
        # context measurement separates positive from negative rotations
        # if the rotation is +pi/4 we will draw 0 w.p. 1
        sim = execute(
            self.spectator_context_qc, backend=BasicAer.get_backend(
                'qasm_simulator'),
            shots=1)
        self.state = np.array(
            list(sim.result().get_counts().keys())
        ).astype(int)

    def _get_reward(self, action):
        correction_theta = (np.pi * np.linspace(-1, 1, self.num_arms))[action]
        # rotations along the same axis commute
        self.update_spectator_circuit(self.spectator_reward_qc,
                                 self.error_samples[self.current_step] + correction_theta)
        sim = execute(
            self.spectator_reward_qc,
            backend=BasicAer.get_backend('qasm_simulator'), shots=1)
        outcome = int(
            list(sim.result().get_counts().keys())[0]
        )

        # fidelity reward
        # debugging only, not observable by agent
        self.info = [
            np.abs(rz(self.error_samples[self.current_step] + correction_theta).tr()) / 2,
            np.abs(rz(self.error_samples[self.current_step]).tr()) / 2
            ]

        # if the correction is perfect then the reward measurement is 0 w.p. 1
        # the underlying distribution we are sampling is monotonic in fidelity
        return 1 if (outcome == 0) else 0

    def render(self, mode='human'):
        pass
    
    def create_spectator_context_circuit(self, error_theta):
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        qc.h(qr)

        qc.rz(error_theta, qr)

        qc.sdg(qr)
        qc.h(qr)
        qc.measure(qr, cr)

        return qc

    def create_spectator_reward_circuit(self, error_theta):
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        qc.h(qr)

        qc.rz(error_theta, qr)

        qc.h(qr)
        qc.measure(qr, cr)

        return qc

    # explicit update function allows us to avoid creating a new ciruit object
    # at every iteration
    def update_spectator_circuit(self, qc, error_theta):
        inst, qarg, carg = qc.data[1]
        qc.data[1] = RZGate(error_theta), qarg, carg

