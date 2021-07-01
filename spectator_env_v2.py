from spectator_env_utils_v2 import (
    create_spectator_analytic_circuit,
    update_spectator_analytic_circuit,
    get_error_unitary
)


import numpy as np

from gym import Env
from gym.spaces import MultiBinary, Box, Tuple

from qiskit import BasicAer, execute
from qutip import rx
from qutip.operators import sigmaz
from qutip import qeye


class SpectatorEnvApi(Env):
    def reset(self):
        pass

    def step(self, actions, feedback_alloc):
        pass

    def set_error_samples(self, new_error_samples):
        pass

    def render(self, mode="human"):
        pass

    def _choose_next_state(self, actions):
        pass

    def _get_reward(self, actions, feedback_alloc):
        pass


class SpectatorEnvBase(SpectatorEnvApi):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_context_spectators: int = 2,
        context_sensitivity: int = 1.0,
        reward_sensitivity: int = 1.0,
    ):
        assert num_context_spectators > 0
        assert batch_size >= 1
        assert len(error_samples) > 0

        self.num_context_spectators = num_context_spectators
        self.error_samples = error_samples

        self.context_sensitivity = context_sensitivity
        self.reward_sensitivity = reward_sensitivity

        self.error_samples_batch = []
        self.batch_size = batch_size

        self.observation_space = MultiBinary(num_context_spectators)

        self.current_step = 0
        self.num_resets = 0

    def reset(self, init_actions):
        self.current_step = 0
        self.num_resets += 1

        self.error_samples_batch = self.error_samples[
            self.current_step: self.current_step + self.batch_size
        ]
        batched_state = self._choose_next_state(init_actions)
        self.current_step += self.batch_size

        return batched_state

    def step(self, actions, feedback_alloc):
        reward, info = self._get_reward(actions, feedback_alloc)

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

    def _get_reward(self, actions, feedback_alloc):
        pass


class SpectatorEnvContinuousV2(SpectatorEnvBase):
    def __init__(
        self,
        error_samples,
        batch_size: int = 1,
        num_context_spectators: int = 2,
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
        # U3(𝜃,𝜙,𝜆) = RZ(𝜙)RX(−𝜋/2)RZ(𝜃)RX(𝜋/2)RZ(𝜆)
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

    def _get_correction(self, t):
        g = [
            1j * t[0] * self.sigmas[0] / 2,
            1j * t[1] * self.sigmas[1] / 2,
            1j * t[2] * self.sigmas[2] / 2
        ]
        return (g[2].expm() * rx(-np.pi / 2) * g[1].expm() * rx(np.pi / 2)
                * g[0].expm())

    # sets batched state
    def _choose_next_state(self, actions):
        assert actions is None or len(actions) == self.batch_size
        context_theta = [action['context'] for action in actions]

        batched_state = []
        for sample, _context_theta in zip(self.error_samples_batch,
                                          context_theta):
            preps = self._get_preps(_context_theta)
            obs = self._get_obs(_context_theta)
            error_unitary = get_error_unitary(
                sample, sensitivity=self.context_sensitivity)
            circuit = update_spectator_analytic_circuit(
                qc=self.spectator_analytic_circuit, error_unitary=error_unitary,
                theta=_context_theta[0], herm=self.sigmas[0], prep=preps[0],
                obs=obs[0],
                parameter_shift=0,
                basis_coin=1
            )

            sim = execute(
                circuit,
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=int(self.num_context_spectators),
                memory=True,
            )

            batched_state.append(np.array(
                sim.result().get_memory()).astype(int))
        return np.array(batched_state)

    def _get_analytic_feedback(self, error_unitary,
                               correction_theta, sigma, prep, obs,
                               num_spectators, parameter_shifts, basis_coin=1):
        feedback = []
        for parameter_shift in parameter_shifts:
            circuit = update_spectator_analytic_circuit(
                qc=self.spectator_analytic_circuit,
                error_unitary=error_unitary, theta=correction_theta,
                herm=sigma, prep=prep, obs=obs,
                parameter_shift=parameter_shift,
                basis_coin=basis_coin)

            sim = execute(
                circuit,
                shots=int(num_spectators),
                backend=BasicAer.get_backend("qasm_simulator"),
                memory=True,
            )

            feedback.append(
                np.array(sim.result().get_memory()).astype(int))
        return feedback

    def _get_reward_correction(self, actions, alloc):
        feedback_set = []
        # Compute gradient per variational param.
        for idx in range(self.num_variational_params):
            feedback = []
            for sample, action in zip(self.error_samples_batch, actions):
                correction = action['correction']
                error_unitary = get_error_unitary(
                    sample, sensitivity=self.reward_sensitivity)

                feedback.append(self._get_analytic_feedback(
                    error_unitary=error_unitary, correction_theta=correction[idx],
                    sigma=self.sigmas[idx],
                    prep=self._get_preps(correction)[idx],
                    obs=self._get_obs(correction)[idx],
                    num_spectators=alloc / 2,
                    parameter_shifts=[-np.pi/2, np.pi/2],
                    basis_coin=np.random.choice([0, 1, 2])))

            feedback_set.append(feedback)
        return feedback_set

    def _get_reward_context(self, actions, alloc):
        feedback_set = []
        # Compute gradient per variational param.
        for idx in range(self.num_variational_params):
            feedback = []
            for sample, action in zip(self.error_samples_batch, actions):
                context_action = action['context']
                error_unitary = get_error_unitary(
                    sample, sensitivity=self.context_sensitivity)

                feedback.append(self._get_analytic_feedback(
                    error_unitary, context_action[idx],
                    self.sigmas[idx],
                    prep=self._get_preps(context_action)[idx],
                    obs=self._get_obs(context_action)[idx],
                    # In this case, we do need all three measurements.
                    num_spectators=alloc / 3,
                    parameter_shifts=[-np.pi/2, 0, np.pi/2]))

            feedback_set.append(feedback)
        return feedback_set

    def _get_reward(self, actions, feedback_alloc):
        context_alloc = feedback_alloc['context']
        correction_alloc = feedback_alloc['correction']
        # Although not strictly necessary, eases the analytic gradients logic.
        # In practice, it is reasonable to split these between error samples
        # unevenly.
        alloc_evenness_msg = "Select a context alloc divisble by "
        assert context_alloc % 3 == 0, alloc_evenness_msg + "3"
        assert correction_alloc % 2 == 0, alloc_evenness_msg + "2"
        assert len(actions) == self.batch_size

        info = []
        for sample, action in zip(self.error_samples_batch, actions):
            correction = action['correction']
            # Not observable by agent (hidden state).
            corr = self._get_correction(np.array(correction))
            # Actual error applied to data qubit.
            error_unitary = get_error_unitary(sample, sensitivity=1.0)
#             print(f"error unitary: {error_unitary}")
#             print(f"corr: {corr}")
            
            control_fid = (np.linalg.norm(error_unitary.tr()) / 2) ** 2
            fid_data = (
                np.linalg.norm((corr * error_unitary).tr()) / 2) ** 2
#             print(f"fid data: {fid_data} =? {max(fid_data, 1-fid_data)}")
            info.append(
                {
#                     'data_fidelity': max(fid_data, 1-fid_data),
#                     'control_fidelity': max(control_fid, 1-control_fid)
                    'data_fidelity': fid_data,
                    'control_fidelity': control_fid,
                }
            )

        return ({'batched_context_feedback': np.array(
                    self._get_reward_context(actions, context_alloc)),
                'batched_correction_feedback': np.array(
                    self._get_reward_correction(actions, correction_alloc))},
                info)


# Optimized action for a conditioning of the overall error distribution.
class Context:
    def __init__(self, gamma, eta, correction_theta_init=[0, 0, 0]):
        # Discount factor.
        # This parameter is deprecated given that we rely on gradient updates
        # to adjust to non-stationarity.
        self.gamma = gamma
        # Gradient step size.
        self.eta = eta

        # Feedback which is batched until we decide to use it.
        self.batch_correction_feedback = ([], [], [])
        self.correction_theta = correction_theta_init

        self.grads = [0, 0, 0]

    def reset(self):
        self.batch_correction_feedback = ([], [], [])

    def discount(self):
        pass

    def update_gamma(self, gamma):
        self.gamma = gamma

    def update_batch_feedback(self, correction_feedback, idx):
        self.batch_correction_feedback[idx].append(correction_feedback)

    def combine_correction_feedback(self):
        # feedback is given per variational param
        for idx, f in enumerate(self.batch_correction_feedback):
            if len(f) == 0:
                continue
            lo = np.array([r[0] for r in f])
            hi = np.array([r[1] for r in f])
            lo = np.array(list(map(lambda x: -1 if x == 0 else 1, lo.flatten())))
            hi = np.array(list(map(lambda x: -1 if x == 0 else 1, hi.flatten())))

            mu_plus = np.mean(hi)
            mu_minus = np.mean(lo)

            grad = mu_plus - mu_minus

            self.correction_theta[idx] -= self.eta * grad
            self.grads[idx] = grad

        self.reset()

    def get_optimal_params(self):
        return self.correction_theta, self.grads


# Optimized action for a conditioning of the overall error distribution.
class IdentityContext(Context):
    def combine_correction_feedback(self):
        self.reset()

    def get_optimal_params(self):
        return [0, 0, 0], [0, 0, 0]


# Contextual analytic geometric descent
class Analytic2D:
    def __init__(self, env, initial_gamma=1.0, context_eta=np.pi/64,
                 correction_eta=np.pi/64, context_theta_init=[0, 0, 0],
                 correction_theta_init=[[0, 0, 0], [0, 0, 0]],
                 identity_threshold=0):
        # Contexts are defined in terms of a function on spectator outcomes.
        # Each context learns an optimal correction independently.
        self.contexts = [Context(initial_gamma, correction_eta,
                                 correction_theta_init[0]),
                         Context(initial_gamma, correction_eta,
                                 correction_theta_init[1]),
                         IdentityContext(initial_gamma, correction_eta)]

        self.num_context_spectators = env.num_context_spectators

        # step size
        self.eta = context_eta

        self.batch_context_feedback = ([], [], [])
        self.context_theta = context_theta_init

        self.grads = [0, 0, 0]

        self.identity_threshold = identity_threshold

    def get_actions(self, observations=None, batch_size=1):
        # Our context is an array of binary spectator qubit measurements.
        # Hence, we could convert this binary array to an integer and index
        # 2^(spectator qubits) contexts.
        # For now, we only have two contexts (+ vs -), and so we consider
        # spectators to be indistinguishable noise polling devices.
        # In the future, we may consider noise gradients and so we do indeed
        # need to track the specific arrangement.

        if observations is None:
            return [{'context': self.context_theta} for i in range(batch_size)]

        actions = []
        for observation in observations:
            if np.abs(np.sum(observation) - self.num_context_spectators / 2) < self.identity_threshold:
                context_idx = 2
            elif np.sum(observation) > self.num_context_spectators / 2:
                context_idx = 1
            else:
                context_idx = 0
            context = self.contexts[context_idx]

            optimal_correction, correction_grad = context.get_optimal_params()
            actions.append(
                {'correction': optimal_correction,
                 'correction_grad': correction_grad,
                 'context_grad': self.grads,
                 'context': self.context_theta})
        return actions

    def combine_correction_feedback(self):
        for context in self.contexts:
            context.combine_correction_feedback()

    def combine_contextual_feedback(self):
        # Feedback is given per gate parameter.
        for idx, f in enumerate(self.batch_context_feedback):
            if len(f) == 0:
                continue
            lo = np.array([r[0] for r in f])
            mid = np.array([r[1] for r in f])
            hi = np.array([r[2] for r in f])

            lo = np.array(list(map(lambda x: -1 if x == 0 else 1,
                                   lo.flatten())))
            mid = np.array(list(map(lambda x: -1 if x == 0 else 1,
                                    mid.flatten())))
            hi = np.array(list(map(lambda x: -1 if x == 0 else 1,
                                   hi.flatten())))

            mean_mid = np.mean(mid)
            mean_lo = np.mean(lo)
            mean_hi = np.mean(hi)
            var_grad = np.mean([2 * (m - mean_mid)
                                * ((h - l) - (mean_hi - mean_lo))
                                for l, m, h in zip(lo, mid, hi)])
            self.grads[idx] = var_grad

            self.context_theta[idx] += self.eta * var_grad
        self.reset()

    def reset(self):
        self.batch_context_feedback = ([], [], [])

    def update_correction_feedback(self, correction_feedback, observations):
        # Feedback is given per variational param.
        for idx in range(len(self.context_theta)):
            for _correction_feedback, observation in zip(
                    correction_feedback[idx], observations):
                if np.abs(np.sum(observation) - self.num_context_spectators / 2) < self.identity_threshold:
                    context_idx = 2
                elif np.sum(observation) > self.num_context_spectators / 2:
                    context_idx = 1
                else:
                    context_idx = 0
                context = self.contexts[context_idx]
                context.update_batch_feedback(_correction_feedback, idx)

    def update_context_feedback(self, context_feedback, observations):
        # Feedback is given per variational param.
        for idx in range(len(self.context_theta)):
            for _context_feedback, observation in zip(
                    context_feedback[idx], observations):
                self.batch_context_feedback[idx].append(_context_feedback)

    def _update_gammas(self, gamma):
        for context in self.contexts:
            context.update_gamma(gamma)