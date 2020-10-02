from spectator_env_utils_v2 import (
    create_spectator_analytic_circuits,
    update_spectator_analytic_circuits
)


import numpy as np

from gym import Env
from gym.spaces import MultiBinary, Box, Tuple

from qiskit import BasicAer, execute
from qutip import rz
from qutip.operators import sigmax, sigmay
from qutip import qeye


class SpectatorEnvApi(Env):
    def reset(self):
        pass

    def step(self, actions):
        pass

    def set_error_samples(self, new_error_samples):
        pass

    def render(self, mode="human"):
        pass

    def _choose_next_state(self, actions):
        pass

    def _get_reward(self, actions):
        pass


class SpectatorEnvBase(SpectatorEnvApi):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_context_spectators: int = 2,
        num_reward_spectators: int = 2,
        context_sensitivity: int = 1.0,
        reward_sensitivity: int = 1.0,
    ):
        assert num_context_spectators > 0
        assert num_reward_spectators > 0
        assert batch_size >= 1
        assert len(error_samples) > 0

        # Although not strictly necessary, eases the analytic gradients logic.
        # We need three circuits per error sample to measure the gradient of
        # interest in expectation. In practice, it is reasonable to split these
        # between error samples unevenly.
        mod_3_msg = "Select a number of spectator divisible by 3"
        assert num_context_spectators % 3 == 0, mod_3_msg
        assert num_reward_spectators % 3 == 0, mod_3_msg

        self.num_context_spectators = num_context_spectators
        self.num_reward_spectators = num_reward_spectators
        self.error_samples = error_samples

        self.context_sensitivity = context_sensitivity
        self.reward_sensitivity = reward_sensitivity

        self.error_samples_batch = []
        self.batch_size = batch_size

        self.observation_space = MultiBinary(num_context_spectators)

        self.current_step = 0
        # Becomes 0 after __init__ exits.
        self.num_resets = -1
        self.reset()

    def reset(self):
        self.current_step = 0
        self.num_resets += 1

        self.error_samples_batch = self.error_samples[
            self.current_step: self.current_step + self.batch_size
        ]
        self._choose_next_state()
        self.current_step += self.batch_size

        return self.batched_state

    def step(self, actions):
        reward = self._get_reward(actions)

        self.error_samples_batch = self.error_samples[
            self.current_step: self.current_step + self.batch_size
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


class SpectatorEnvContinuousV2(SpectatorEnvBase):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_context_spectators: int = 2,
        num_reward_spectators: int = 2,
        context_sensitivity: int = 1.0,
        reward_sensitivity: int = 1.0,
    ):
        self.action_space = Tuple(
            (
                # correction thetas
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
                # context thetas
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
                Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32)
            )
        )

        self.sigmas = [
            sigmay(),
            sigmax(),
            sigmay()
            ]

        self.spectator_circuit_sets = [
            create_spectator_analytic_circuits(error_unitary=qeye(2), theta=0,
                                               herm=self.sigmas[0], prep=qeye(2),obs=qeye(2)),
            create_spectator_analytic_circuits(error_unitary=qeye(2), theta=0,
                                               herm=self.sigmas[1], prep=qeye(2), obs=qeye(2)),
            create_spectator_analytic_circuits(error_unitary=qeye(2), theta=0,
                                               herm=self.sigmas[2], prep=qeye(2), obs=qeye(2))
        ]

        super().__init__(
            error_samples,
            batch_size,
            num_context_spectators,
            num_reward_spectators,
            context_sensitivity,
            reward_sensitivity
        )

    def _get_preps(self, t):
        g = [
            1j * t[0] * self.sigmas[0],
            1j * t[1] * self.sigmas[1],
            1j * t[2] * self.sigmas[2]
        ]
        return [
            qeye(2),
            g[0].expm(),
            g[1].expm() * g[0].expm()
        ]

    def _get_obs(self, t):
        g = [
            1j * t[0] * self.sigmas[0],
            1j * t[1] * self.sigmas[1],
            1j * t[2] * self.sigmas[2]
        ]
        return [
            g[2].expm() * g[1].expm(),
            g[2].expm(),
            qeye(2)
        ]

    def _get_error_unitary(self, sample):
        return rz(sample)

    def _get_correction(self, t):
        g = [
            1j * t[0] * self.sigmas[0],
            1j * t[1] * self.sigmas[1],
            1j * t[2] * self.sigmas[2]
        ]
        return g[2].expm() * g[1].expm() * g[0].expm()

    # sets batched state
    def _choose_next_state(self, actions=None):
        batched_state = []
        context_theta = (
            np.repeat([[0, 0, 0]], self.batch_size, axis=0)
            if actions is None
            else [action['context'] for action in actions]
        )

        circuit_set = self.spectator_circuit_sets[0]
        for sample, _context_theta in zip(self.error_samples_batch, context_theta):
            preps = self._get_preps(_context_theta)
            obs = self._get_obs(_context_theta)
            error_unitary = self._get_error_unitary(sample)
            # self.context_sensitivity
            update_spectator_analytic_circuits(
                circuit_set, error_unitary, _context_theta[0], self.sigmas[0], preps[0], obs[0]
            )
            sim = execute(
                circuit_set[1],
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=self.num_context_spectators,
                memory=True,
            )

            batched_state.append(np.array(sim.result().get_memory()).astype(int))
        self.batched_state = np.array(batched_state)

    def _get_analytic_feedback(self, circuit_set, error_unitary,
                               correction_theta, sigma, prep, obs,
                               num_spectators, sensitivity):
        circuit_set = update_spectator_analytic_circuits(
            circuit_set, error_unitary, correction_theta, sigma, prep, obs
        )

        f = []
        for circuit in circuit_set:
            sim = execute(
                circuit,
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=num_spectators,
                memory=True,
            )

            f.append(
                np.array(sim.result().get_memory()).astype(int))
        return f

    def _get_reward(self, actions):
        info = []
        context_feedback_set = []
        correction_feedback_set = []
        for idx, circuit_set in enumerate(self.spectator_circuit_sets):
            info = []
            context_feedback = []
            correction_feedback = []
            for sample, action in zip(self.error_samples_batch, actions):
                correction_theta = action['correction']
                context_theta = action['context']
                error_unitary = self._get_error_unitary(sample)

                correction_feedback.append(self._get_analytic_feedback(
                    circuit_set, error_unitary, correction_theta[idx],
                    self.sigmas[idx], self._get_preps(correction_theta)[idx],
                    self._get_preps(correction_theta)[idx],
                    self.num_reward_spectators, self.reward_sensitivity))

                context_feedback.append(self._get_analytic_feedback(
                    circuit_set, error_unitary, context_theta[idx],
                    self.sigmas[idx], self._get_preps(context_theta)[idx],
                    self._get_preps(context_theta)[idx],
                    self.num_reward_spectators, self.context_sensitivity))

                # not observable by agent (hidden state)
                corr = self._get_correction(
                    np.array(correction_theta) / self.reward_sensitivity)
                info.append(
                    [
                        np.abs((corr * error_unitary).tr()) / 2,
                        np.abs(error_unitary.tr()) / 2,
                    ]
                )
            context_feedback_set.append(context_feedback)
            correction_feedback_set.append(correction_feedback)

        self.info = info
        return {'batched_context_feedback': np.array(context_feedback_set),
                'batched_correction_feedback': np.array(correction_feedback_set)}
