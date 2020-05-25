from spectator_env_utils import (
    create_spectator_context_circuit,
    create_spectator_reward_circuit,
    update_spectator_circuit,
)

import numpy as np

from gym import Env
from gym.spaces import Discrete, MultiBinary, Box, Tuple

from qiskit import BasicAer, execute
from qutip import rz


class SpectatorEnvBase(Env):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_context_spectators: int = 2,
        num_reward_spectators: int = 2,
    ):
        assert num_context_spectators >= 2
        assert num_reward_spectators >= 2
        assert batch_size >= 1
        assert len(error_samples) > 0

        self.spectator_context_qc = create_spectator_context_circuit(0)
        self.spectator_reward_qc = create_spectator_reward_circuit(0)

        self.num_context_spectators = num_context_spectators
        self.num_reward_spectators = num_reward_spectators
        self.error_samples = error_samples

        self.error_samples_batch = []
        self.batch_size = batch_size

        self.observation_space = MultiBinary(num_context_spectators)

        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self):
        self.current_step = 0
        self.num_resets += 1

        self.error_samples_batch = self.error_samples[
            self.current_step : self.current_step + self.batch_size
        ]
        self._choose_next_state()
        self.current_step += self.batch_size

        return self.batched_state

    def step(self, actions):
        reward = self._get_reward(actions)

        self.error_samples_batch = self.error_samples[
            self.current_step : self.current_step + self.batch_size
        ]
        self._choose_next_state(actions)
        self.current_step += self.batch_size

        done = (
            False
            if self.current_step + self.batch_size < len(self.error_samples)
            else True
        )
        return self.batched_state, reward, done, self.info

    def set_error_samples(self, new_error_samples):
        self.error_samples = new_error_samples

    def render(self, mode="human"):
        pass

    def _choose_next_state(self, actions):
        pass

    def _get_reward(self, actions):
        pass


class SpectatorEnvDiscrete(SpectatorEnvBase):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_arms: int = 21,
        num_context_spectators: int = 2,
        num_reward_spectators: int = 2,
    ):
        assert num_arms >= 1
        super().__init__(
            error_samples, batch_size, num_context_spectators, num_reward_spectators
        )
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
        self.num_arms = num_arms

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


class SpectatorEnvContinuous(SpectatorEnvBase):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_context_spectators: int = 2,
        num_reward_spectators: int = 2,
    ):
        super().__init__(
            error_samples,
            batch_size,
            num_context_spectators,
            num_reward_spectators,
        )
        self.action_space = Tuple(
            (
                # uniform range of rotations to consider for correction
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
                # smoothing delta for gradient approximation
                Box(low=0, high=np.pi, shape=(), dtype=np.float32),
                # contextual measurement rotational bias
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
            )
        )

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
            correction_theta, delta, rotational_bias = action

            # rotations along the same axis commute
            update_spectator_circuit(
                self.spectator_reward_qc, sample + correction_theta - delta + rotational_bias
            )
            sim_lo = execute(
                self.spectator_reward_qc,
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=self.num_reward_spectators // 2,
                memory=True,
            )
            outcome_lo = np.array(sim_lo.result().get_memory()).astype(int)

            update_spectator_circuit(
                self.spectator_reward_qc, sample + correction_theta + delta + rotational_bias
            )
            sim_hi = execute(
                self.spectator_reward_qc,
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=self.num_reward_spectators // 2,
                memory=True,
            )
            outcome_hi = np.array(sim_hi.result().get_memory()).astype(int)

            # fidelity reward
            # debugging only, not observable by agent
            info.append(
                [
                    np.abs(rz(sample + correction_theta + rotational_bias).tr()) / 2,
                    np.abs(rz(sample).tr()) / 2,
                ]
            )
            rewards.append((self.num_reward_spectators // 2 - np.sum(outcome_lo), self.num_reward_spectators // 2 - np.sum(outcome_hi)))

        self.info = np.array(info)
        # if the correction is perfect then the reward measurement is 0 w.p. 1
        # the underlying distribution we are sampling from is fidelity
        return np.array(rewards)
