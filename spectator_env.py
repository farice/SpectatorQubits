from spectator_env_utils import *

import numpy as np
from typing import Optional

from gym import Env, Space
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box, Tuple

from qiskit import Aer, BasicAer, execute
import matplotlib.pyplot as plt
import numpy as np
from qutip import *


class SpectatorEnv(Env):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_arms: int = 21,
        num_context_spectators: int = 1,
        num_reward_spectators: int = 1,
    ):

        self.spectator_context_qc = create_spectator_context_circuit(0)
        self.spectator_reward_qc = create_spectator_reward_circuit(0)

        self.num_arms = num_arms
        self.num_context_spectators = num_context_spectators
        self.num_reward_spectators = num_reward_spectators
        self.error_samples = error_samples

        self.error_samples_batch = []
        self.batch_size = batch_size

        self.action_space = Tuple(
            (
                # ordered arm within uniform rotation action range
                Discrete(num_arms),
                # uniform range of rotations to consider for correction
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
                # contextual measurement rotational bias
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
            )
        )
        self.observation_space = MultiBinary(num_context_spectators)
        self.ep_length = len(error_samples)
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self):
        self.current_step = 0
        self.num_resets += 1

        self._set_error_samples_batch()
        self._choose_next_state()
        self._update_current_step()

        return self.batched_state

    def step(self, actions):
        reward = self._get_reward(actions)

        self._set_error_samples_batch()
        self._choose_next_state(actions)
        self._update_current_step()

        done = (
            False
            if self.current_step + self.batch_size < len(self.error_samples)
            else True
        )
        return self.batched_state, reward, done, self.info

    def set_error_samples(self, new_error_samples):
        self.error_samples = new_error_samples

    def _set_error_samples_batch(self):
        self.error_samples_batch = self.error_samples[
            self.current_step : self.current_step + self.batch_size
        ]

    def _update_current_step(self):
        self.current_step += self.batch_size

    # sets batched state
    def _choose_next_state(self, actions=None):
        rotational_biases = (
            np.zeros(self.batch_size)
            if actions is None
            else [list(action)[2] for action in actions]
        )

        batched_state = []
        for sample, rotational_bias in zip(self.error_samples_batch, rotational_biases):
            update_spectator_circuit(
                self.spectator_context_qc, sample + rotational_bias
            )
            # context measurement separates positive from negative rotations
            # if the rotation is +pi/4 we will draw 0 w.p. 1
            sim = execute(
                self.spectator_context_qc,
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=self.num_context_spectators,
                memory=True,
            )

            batched_state.append(np.array(sim.result().get_memory()).astype(int))

        self.batched_state = np.array(batched_state)

    def _get_reward(self, actions):
        info = []
        rewards = []
        for sample, action in zip(self.error_samples_batch, actions):
            arm, uniform_theta_width, rotational_bias = action
            correction_theta = np.linspace(
                -uniform_theta_width / 2, uniform_theta_width / 2, self.num_arms
            )[arm]
            # print("correction_theta: ", correction_theta)

            # rotations along the same axis commute
            update_spectator_circuit(
                self.spectator_reward_qc, sample + correction_theta + rotational_bias
            )
            sim = execute(
                self.spectator_reward_qc,
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=self.num_reward_spectators,
                memory=True,
            )
            outcome = np.array(sim.result().get_memory()).astype(int)

            # fidelity reward
            # debugging only, not observable by agent
            info.append(
                [
                    np.abs(rz(sample + correction_theta + rotational_bias).tr()) / 2,
                    np.abs(rz(sample).tr()) / 2,
                ]
            )
            rewards.append(self.num_reward_spectators - np.sum(outcome))

        self.info = np.array(info)
        # if the correction is perfect then the reward measurement is 0 w.p. 1
        # the underlying distribution we are sampling from is fidelity
        return np.array(rewards)

    def render(self, mode="human"):
        pass
