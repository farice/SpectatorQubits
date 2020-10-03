from spectator_env_utils_v2 import (
    create_spectator_analytic_circuit,
    update_spectator_analytic_circuit
)


import numpy as np

from gym import Env
from gym.spaces import MultiBinary, Box, Tuple

from qiskit import BasicAer, execute
from qutip import rz, rx
from qutip.operators import sigmaz
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
        batched_state = self._choose_next_state()
        self.current_step += self.batch_size

        return batched_state

    def step(self, actions):
        reward, info = self._get_reward(actions)

        self.error_samples_batch = self.error_samples[
            self.current_step: self.current_step + self.batch_size
        ]
        batched_state = self._choose_next_state(actions)
        self.current_step += self.batch_size

        done = (
            False
            if self.current_step + self.batch_size < len(self.error_samples)
            else True
        )
        return batched_state, reward, done, info

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
        num_context_spectators: int = 3,
        num_reward_spectators: int = 3,
        context_sensitivity: int = 1.0,
        reward_sensitivity: int = 1.0,
        # single qubit unitary can be parameterized in Euler rotation basis
        num_variational_params: int = 3,
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

        # We parameterize our single-qubit rotation by a U3 gate
        # 𝑈3(𝜃,𝜙,𝜆)=𝑅𝑍(𝜙)𝑅𝑋(−𝜋/2)𝑅𝑍(𝜃)𝑅𝑋(𝜋/2)𝑅𝑍(𝜆)
        self.sigmas = [
            sigmaz(),
            sigmaz(),
            sigmaz()
            ]

        self.num_variational_params = num_variational_params

        # This is the general circuit object which is reused for the various
        # context/reward measurements. The initialization is arbitrary.
        self.spectator_analytic_circuit = create_spectator_analytic_circuit(
            error_unitary=qeye(2), theta=0, herm=qeye(2), prep=qeye(2),
            obs=qeye(2), parameter_shift=0)

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
            1j * t[0] * self.sigmas[0] / 2,
            1j * t[1] * self.sigmas[1] / 2,
            1j * t[2] * self.sigmas[2] / 2
        ]
        return [
            qeye(2),
            rx(np.pi / 2) * g[0].expm(),
            rx(-np.pi / 2) * g[1].expm() * rx(np.pi/2) * g[0].expm()
        ]

    def _get_obs(self, t):
        g = [
            1j * t[0] * self.sigmas[0] / 2,
            1j * t[1] * self.sigmas[1] / 2,
            1j * t[2] * self.sigmas[2] / 2
        ]
        return [
            g[2].expm() * rx(-np.pi / 2) * g[1].expm() * rx(np.pi / 2),
            g[2].expm() * rx(-np.pi / 2),
            qeye(2)
        ]

    def _get_error_unitary(self, sample):
        return rz(sample)

    def _get_correction(self, t):
        g = [
            1j * t[0] * self.sigmas[0] / 2 ,
            1j * t[1] * self.sigmas[1] / 2,
            1j * t[2] * self.sigmas[2] / 2
        ]
        return g[2].expm() * rx(-np.pi / 2) * g[1].expm() * rx(np.pi / 2) * g[0].expm()

    def _get_analytic_feedback(self, error_unitary,
                               correction_theta, sigma, prep, obs,
                               num_spectators, sensitivity):
        feedback = []
        for parameter_shift in [-np.pi/2, 0, np.pi/2]:
            circuit = update_spectator_analytic_circuit(
                qc=self.spectator_analytic_circuit,
                error_unitary=error_unitary, theta=correction_theta,
                herm=sigma, prep=prep, obs=obs,
                parameter_shift=parameter_shift)
#             print("feedback", parameter_shift, correction_theta, error_unitary)
#             print(circuit)
            sim = execute(
                circuit,
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=num_spectators,
                memory=True,
            )

            feedback.append(
                np.array(sim.result().get_memory()).astype(int))
        return feedback

    # sets batched state
    def _choose_next_state(self, actions=None):
        assert actions is None or len(actions) == self.batch_size
        context_theta = (
            np.repeat([[0, 0, 0]], self.batch_size, axis=0)
            if actions is None
            else [action['context'] for action in actions]
        )

        batched_state = []
        for sample, _context_theta in zip(self.error_samples_batch,
                                          context_theta):
            preps = self._get_preps(_context_theta)
            obs = self._get_obs(_context_theta)
            error_unitary = self._get_error_unitary(sample)
            # self.context_sensitivity
            circuit = update_spectator_analytic_circuit(
                qc=self.spectator_analytic_circuit, error_unitary=error_unitary,
                theta=_context_theta[0], herm=self.sigmas[0], prep=preps[0], obs=obs[0],
                parameter_shift=0
            )
#             print("context", _context_theta, sample)
#             print(circuit)
            sim = execute(
                circuit,
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=self.num_context_spectators,
                memory=True,
            )

            batched_state.append(np.array(sim.result().get_memory()).astype(int))
        return np.array(batched_state)

    def _get_reward(self, actions):
        assert len(actions) == self.batch_size
        info = []
        context_feedback_set = []
        correction_feedback_set = []
        # compute gradient per variational param
        for idx in range(self.num_variational_params):
            info = []
            context_feedback = []
            correction_feedback = []
            for sample, action in zip(self.error_samples_batch, actions):
                correction_theta = action['correction']
                context_theta = action['context']
                error_unitary = self._get_error_unitary(sample)

                correction_feedback.append(self._get_analytic_feedback(
                    error_unitary=error_unitary, correction_theta=correction_theta[idx],
                    sigma=self.sigmas[idx], prep=self._get_preps(correction_theta)[idx],
                    obs=self._get_obs(correction_theta)[idx],
                    num_spectators=self.num_reward_spectators,
                    sensitivity=self.reward_sensitivity))

                context_feedback.append(self._get_analytic_feedback(
                    error_unitary, context_theta[idx],
                    self.sigmas[idx], prep=self._get_preps(context_theta)[idx],
                    obs=self._get_obs(context_theta)[idx],
                    num_spectators=self.num_reward_spectators,
                    sensitivity=self.context_sensitivity))

                # not observable by agent (hidden state)
                corr = self._get_correction(
                    np.array(correction_theta) / self.reward_sensitivity)
                fid = (np.linalg.norm((corr.conj() * error_unitary).tr()) / 2) ** 2
                control_fid = (np.linalg.norm(error_unitary.tr()) / 2) ** 2
                info.append(
                    [
                        fid,
                        control_fid,
                    ]
                )
            context_feedback_set.append(context_feedback)
            correction_feedback_set.append(correction_feedback)

        return {'batched_context_feedback': np.array(context_feedback_set),
                'batched_correction_feedback': np.array(
                    correction_feedback_set)}, info
