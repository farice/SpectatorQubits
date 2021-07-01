from dataclasses import dataclass
from typing import List, Any

import numpy as np

from spectator_env_v2 import SpectatorEnvContinuousV2, Analytic2D
from spectator_env_utils_v2 import extract_theta_phi


@dataclass
class ParallelSimResult:
    done: bool
    data_fidelity_per_episode: List[Any]
    control_fidelity_per_episode: List[Any]
    context_2d_repr: List[Any]
    correction_2d_repr: List[Any]


class ParallelSim:
    def __init__(self, error_samples_generator,
                 feedback_spectators_allocation_function, context_eta=np.pi/16,
                 correction_eta=np.pi/16, context_eta_init=[0, 0, 0],
                 correction_eta_init=[0, 0, 0], num_context_spectators=7,
                 batch_size=1, num_batches_to_combine_correction_feedback=1,
                 num_batches_to_combine_context_feedback=1, identity_threshold=0):
        # Should store all hyperparams so that config can be accessed readily in analysis.
        self.context_eta = context_eta
        self.correction_eta = correction_eta
        self.context_eta_init = context_eta_init
        self.correction_eta_init = correction_eta_init
        self.batch_size = batch_size
        self.num_context_spectators = num_context_spectators
        self.error_samples_generator = error_samples_generator
        self.feedback_spectators_allocation_function = feedback_spectators_allocation_function
        self.num_batches_to_combine_correction_feedback = num_batches_to_combine_correction_feedback
        self.num_batches_to_combine_context_feedback = num_batches_to_combine_context_feedback

        error_samples = error_samples_generator(iteration=0, batch_size=self.batch_size)
        self.env = SpectatorEnvContinuousV2(
            error_samples, batch_size=batch_size,
            num_context_spectators=num_context_spectators)
        self.md = Analytic2D(self.env, context_eta=context_eta,
                             correction_eta=correction_eta,
                             context_theta_init=context_eta_init,
                             correction_theta_init=correction_eta_init,
                             identity_threshold=identity_threshold)
        self.data_fidelity_per_episode = []
        self.control_fidelity_per_episode = []
        self.data_fidelity = []
        self.control_fidelity = []
        self.correction_2d_repr = {}
        self.context_2d_repr = []
        self.observation = self.env.reset(self.md.get_actions(
                                          batch_size=batch_size))
        self.frame_idx = 0

    def destruct(self):
        self.error_samples_generator = None
        self.feedback_spectators_allocation_function = None
        self.env = None
        self.md = None

    def set_error_samples(self, new_error_samples):
        self.env.set_error_samples(new_error_samples)

    def step(self):
        '''
        Beginning of main logic.
        '''
        
#         print(f"idx: {self.frame_idx}")
#         print(f"obs: {self.observation}")
#         print(f"error samples: {self.env.error_samples}")
        # This will return the known optimal actions as a function of the context
        # mapping.
        actions = self.md.get_actions(self.observation,
                                      batch_size=self.batch_size)
        prev_observation = self.observation

        # Allocation of non-contextual spectators between optimizing the objectives.
        feedback_alloc = self.feedback_spectators_allocation_function()
        self.observation, feedback, done, info = self.env.step(actions, feedback_alloc)
        self.observation = None if done else self.observation

        self.md.update_correction_feedback(
                  correction_feedback=feedback['batched_correction_feedback'],
                  observations=prev_observation)
        self.md.update_context_feedback(
                context_feedback=feedback['batched_context_feedback'],
                observations=prev_observation)

        if self.frame_idx % self.num_batches_to_combine_correction_feedback == 0:
            self.md.combine_correction_feedback()
        if self.frame_idx % self.num_batches_to_combine_context_feedback == 0:
            self.md.combine_contextual_feedback()

        new_error_samples = self.error_samples_generator(iteration=self.frame_idx)
        self.set_error_samples(new_error_samples)
        
#         print("\n")
        '''
        End of main logic.
        '''

        for _info in info:
#             print(f"data fid: {_info['data_fidelity']}")
            self.data_fidelity.append(_info['data_fidelity'])
            self.control_fidelity.append(_info['control_fidelity'])

        self.context_2d_repr.append(extract_theta_phi(
            self.env._get_correction(self.md.context_theta).dag()))
        for idx, c in enumerate(self.md.contexts[:2]):
            correction_2d = extract_theta_phi(
                    self.env._get_correction(c.correction_theta).dag())
            if idx in self.correction_2d_repr.keys():
                self.correction_2d_repr[idx].append(correction_2d)
            else:
                self.correction_2d_repr[idx] = [correction_2d]

        if done:
            self.data_fidelity_per_episode.append(np.mean(self.data_fidelity))
            self.control_fidelity_per_episode.append(np.mean(self.control_fidelity))
            self.data_fidelity = []
            self.control_fidelity = []

            self.observation = self.env.reset(actions)

        self.frame_idx += 1
        return ParallelSimResult(
            done=done,
            data_fidelity_per_episode=self.data_fidelity_per_episode,
            control_fidelity_per_episode=self.control_fidelity_per_episode,
            context_2d_repr=self.context_2d_repr,
            correction_2d_repr=self.correction_2d_repr
        )
