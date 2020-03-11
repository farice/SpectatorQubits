import numpy as np
from typing import Optional

from gym import Env, Space
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, BasicAer, execute
from qiskit.extensions.standard.rz import RZGate
import matplotlib.pyplot as plt
import numpy as np
from qutip import *


class SpectatorEnv(Env):
    def __init__(self,
                 error_samples,
                 num_arms: int = 21,
                 num_spectators: int = 1):

        self.spectator_context_qc = self.create_spectator_context_circuit(0)
        self.spectator_reward_qc = self.create_spectator_reward_circuit(0)

        self.num_arms = num_arms
        self.num_spectators = num_spectators
        self.error_samples = error_samples

        self.action_space = Tuple((
            # ordered arm within uniform rotation action range
            Discrete(num_arms), 
            # uniform range of rotations to consider for correction
            Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
            # contextual measurement rotational bias
            Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32)
            )
            )
        self.observation_space = MultiBinary(num_spectators)
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
        if self.current_step + 1 < len(self.error_samples):
            self._choose_next_state(action)
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, self.info

    def _choose_next_state(self, action=None):
        rotational_bias = 0 if action is None else list(action)[2]
        self.update_spectator_circuit(self.spectator_context_qc, self.error_samples[self.current_step + 1] + rotational_bias)
        # context measurement separates positive from negative rotations
        # if the rotation is +pi/4 we will draw 0 w.p. 1
        sim = execute(
            self.spectator_context_qc, backend=BasicAer.get_backend(
                'qasm_simulator'),
            shots=self.num_spectators, memory=True)
        self.state = np.array(
            sim.result().get_memory()
        ).astype(int)

    def _get_reward(self, action):
        arm, uniform_theta_width, rotational_bias = action
        correction_theta = np.linspace(-uniform_theta_width / 2, uniform_theta_width / 2, self.num_arms)[arm]
#         print("correction_theta: ", correction_theta)
#         print("error sample: ", self.error_samples[self.current_step])
        # rotations along the same axis commute
        self.update_spectator_circuit(self.spectator_reward_qc,
                                 self.error_samples[self.current_step] + correction_theta + rotational_bias)
        sim = execute(
            self.spectator_reward_qc,
            backend=BasicAer.get_backend('qasm_simulator'), shots=self.num_spectators, memory=True)
        outcome = np.array(
            sim.result().get_memory()
        ).astype(int)

        # fidelity reward
        # debugging only, not observable by agent
        self.info = [
            np.abs(rz(self.error_samples[self.current_step] + correction_theta + rotational_bias).tr()) / 2,
            np.abs(rz(self.error_samples[self.current_step]).tr()) / 2
            ]

        # if the correction is perfect then the reward measurement is 0 w.p. 1
        # the underlying distribution we are sampling is monotonic in fidelity
        return 1 - np.sum(outcome) / self.num_spectators # something like a point estimate of fidelity

    def render(self, mode='human'):
        pass
    
    @staticmethod
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

    @staticmethod
    def create_spectator_reward_circuit(error_theta):
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
    @staticmethod
    def update_spectator_circuit(qc, error_theta):
        inst, qarg, carg = qc.data[1]
        qc.data[1] = RZGate(error_theta), qarg, carg

