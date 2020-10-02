from spectator_env_utils import (
    create_spectator_context_circuit,
    create_spectator_reward_circuit,
    update_spectator_circuit,
)

from spectator_env_utils_v2 import (
    create_spectator_analytic_circuits,
    update_spectator_analytic_circuits
)


import numpy as np

from gym import Env
from gym.spaces import Discrete, MultiBinary, Box, Tuple

from qiskit import BasicAer, execute
from qutip import rz
from qutip.operators import sigmax, sigmay, sigmaz
from qutip.qip.gates import rotation
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
    ):
        assert num_context_spectators >= 2
        assert num_reward_spectators >= 2
        assert batch_size >= 1
        assert len(error_samples) > 0

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

         # self.spectator_context_qc = create_spectator_context_circuit(0, 0, 0, 0, [0,1,0] / np.sqrt(3))
        self.spectator_context_qc = create_spectator_context_circuit(0, 0, 0, np.pi / 2, -np.pi, -np.pi / 2)
        self.spectator_reward_qc = create_spectator_reward_circuit(0, 0, 0, 0, [0,0,1] / np.sqrt(3))

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
                self.spectator_context_qc, 0, 0, sample + rotational_bias
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
                self.spectator_reward_qc, 0, 0, sample, correction_theta + rotational_bias, [0, 0, 1]
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
        context_sensitivity = 1.0,
        reward_sensitivity = 1.0,
    ):

         # self.spectator_context_qc = create_spectator_context_circuit(0, 0, 0, 0, [0,1,0] / np.sqrt(3))
        self.spectator_context_qc = create_spectator_context_circuit(0, 0, 0, np.pi / 2, -np.pi, -np.pi / 2)
        self.spectator_reward_qc = create_spectator_reward_circuit(0, 0, 0, 0, [0,0,1] / np.sqrt(3))

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
        self.context_sensitivity = context_sensitivity
        self.reward_sensitivity = reward_sensitivity
        super().__init__(
            error_samples,
            batch_size,
            num_context_spectators,
            num_reward_spectators,
        )


    # sets batched state
    def _choose_next_state(self, actions=None):
        rotational_biases = (
            np.repeat([[0, 0, 0]], self.batch_size, axis=0)
            if actions is None
            else [list(action)[3] for action in actions]
        )
        batched_state = []
        for sample, rotational_bias in zip(self.error_samples_batch, rotational_biases):
            update_spectator_circuit(
                self.spectator_context_qc, 0, 0, self.context_sensitivity * sample + rotational_bias[2]
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
            correction_vec, direction, delta, _ = action

            # rotations along the same axis commute
            update_spectator_circuit(
                self.spectator_reward_qc, 0, 0, self.reward_sensitivity * sample, np.linalg.norm(correction_vec - delta * direction),
                (correction_vec - delta * direction) / np.linalg.norm(correction_vec - delta * direction)
            )
            sim_lo = execute(
                self.spectator_reward_qc,
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=self.num_reward_spectators // 2,
                memory=True,
            )
            outcome_lo = np.array(sim_lo.result().get_memory()).astype(int)

            update_spectator_circuit(
                self.spectator_reward_qc, 0, 0, self.reward_sensitivity * sample, np.linalg.norm(correction_vec + delta * direction),
                (correction_vec + delta * direction) / np.linalg.norm(correction_vec + delta * direction)
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
            rot_vec = correction_vec[0] * sigmax() + correction_vec[1] * sigmay() + correction_vec[2] * sigmaz()
            info.append(
                [
                    np.abs((rotation(
                    rot_vec / np.linalg.norm(rot_vec),
                    np.linalg.norm(rot_vec)) * rz(self.reward_sensitivity * sample)).tr()) / 2,
                    np.abs(rz(self.reward_sensitivity * sample).tr()) / 2,
                ]
            )
            rewards.append((self.num_reward_spectators // 2 - np.sum(outcome_lo), self.num_reward_spectators // 2 - np.sum(outcome_hi)))

        self.info = np.array(info)
        # if the correction is perfect then the reward measurement is 0 w.p. 1
        # the underlying distribution we are sampling from is fidelity
        return np.array(rewards)


class SpectatorEnvContinuous2d(SpectatorEnvApi):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_context_spectators: int = 2,
        num_reward_spectators: int = 2,
    ):
        self.axis1 = SpectatorEnvContinuous(error_samples[0], batch_size, num_context_spectators)
        self.axis2 = SpectatorEnvContinuous(error_samples[1], batch_size, num_context_spectators)

    def reset(self):
        return self.axis1.reset(), self.axis2.reset()

    def step(self, actions):
        return list(zip(self.axis1.step(actions[0]), self.axis2.step(actions[1])))

    def set_error_samples(self, new_error_samples):
        self.axis1.set_error_samples(new_error_samples[0])
        self.axis2.set_error_samples(new_error_samples[1])

    def _choose_next_state(self, actions):
        self.axis1._choose_next_state(actions[0])
        self.axis2._choose_next_state(actions[1])

    def _get_reward(self, actions):
        self.axis1._get_reward(actions[0])
        self.axis2._get_reward(actions[1])


class SpectatorEnvContinuousV2(SpectatorEnvBase):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_context_spectators: int = 2,
        num_reward_spectators: int = 2,
        context_sensitivity = 1.0,
        reward_sensitivity = 1.0,
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
        self.context_sensitivity = context_sensitivity
        self.reward_sensitivity = reward_sensitivity

        self.sigmas = [
            sigmay(),
            sigmax(),
            sigmay()
            ]

        self.spectator_circuit_sets = [
            create_spectator_analytic_circuits(0, 0, 0, 0, self.sigmas[0], qeye(2), qeye(2)),
            create_spectator_analytic_circuits(0, 0, 0, 0, self.sigmas[1], qeye(2), qeye(2)),
            create_spectator_analytic_circuits(0, 0, 0, 0, self.sigmas[2], qeye(2), qeye(2))
        ]

        super().__init__(
            error_samples,
            batch_size,
            num_context_spectators,
            num_reward_spectators,
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
            else [list(action)[1] for action in actions]
        )

        circuit_set = self.spectator_circuit_sets[0]
        for sample, _context_theta in zip(self.error_samples_batch, context_theta):
            preps = self._get_preps(_context_theta)
            obs = self._get_obs(_context_theta)
            update_spectator_analytic_circuits(
                circuit_set, 0, 0, self.context_sensitivity * sample, _context_theta[0], self.sigmas[0], preps[0], obs[0]
            )
            sim = execute(
                circuit_set[1],
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=self.num_context_spectators,
                memory=True,
            )

            batched_state.append(np.array(sim.result().get_memory()).astype(int))
        self.batched_state = np.array(batched_state)

    def _get_reward(self, actions):
        info = []
        context_feedback_set = []
        correction_feedback_set = []
        for idx, circuit_set in enumerate(self.spectator_circuit_sets):
            info = []
            context_feedback = []
            correction_feedback = []
            for sample, action in zip(self.error_samples_batch, actions):
                correction_theta, context_theta = action

                preps = self._get_preps(correction_theta)
                obs = self._get_obs(correction_theta)

                circuit_set = update_spectator_analytic_circuits(
                    circuit_set, 0, 0, self.reward_sensitivity * sample, correction_theta[idx], self.sigmas[idx], preps[idx], obs[idx]
                )

                f = []
                for circuit in circuit_set:
                    sim = execute(
                        circuit,
                        backend=BasicAer.get_backend("qasm_simulator"),
                        shots=self.num_reward_spectators,
                        memory=True,
                    )

                    f.append(
                    np.array(sim.result().get_memory()).astype(int))
                correction_feedback.append(f)

                preps = self._get_preps(context_theta)
                obs = self._get_obs(context_theta)

                circuit_set = update_spectator_analytic_circuits(
                    circuit_set, 0, 0, self.context_sensitivity * sample, context_theta[idx], self.sigmas[idx], preps[idx], obs[idx]
                )
                f = []
                for circuit in circuit_set:
                    sim = execute(
                        circuit,
                        backend=BasicAer.get_backend("qasm_simulator"),
                        shots=self.num_reward_spectators,
                        memory=True,
                    )

                    f.append(
                        np.array(sim.result().get_memory()).astype(int)
                    )
                context_feedback.append(f)

                # fidelity reward
                # debugging only, not observable by agent
                corr = self._get_correction(np.array(correction_theta) / self.reward_sensitivity)
                info.append(
                    [
                        np.abs((corr * rz(sample)).tr()) / 2,
                        np.abs(rz(sample).tr()) / 2,
                    ]
                )
            context_feedback_set.append(context_feedback)
            correction_feedback_set.append(correction_feedback)

        self.info = info
        # if the correction is perfect then the reward measurement is 0 w.p. 1
        # the underlying distribution we are sampling from is fidelity
        return np.array(context_feedback_set), np.array(correction_feedback_set)
