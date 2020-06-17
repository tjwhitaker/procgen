import numpy as np
import copy
from random import random
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID


class PseudoEnsembleAgent(PPOTrainer):
    def __init__(self, config=None, env=None, logger_creator=None):
        super().__init__(config, env, logger_creator)

    def compute_action(self,
                       observation,
                       state=None,
                       prev_action=None,
                       prev_reward=None,
                       info=None,
                       policy_id=DEFAULT_POLICY_ID,
                       full_fetch=False,
                       explore=None):

        # Create pseudo ensemble actions
        ensemble_actions = []
        og_weights = self.get_policy().get_weights()

        for i in range(8):
            new_weights = self.prune_weights(og_weights, 0.1)
            self.get_policy().set_weights(new_weights)
            ensemble_actions.append(super().compute_action(observation, state, prev_action,
                                                           prev_reward, info, policy_id, full_fetch, explore))

        # Reset Weights
        self.get_policy().set_weights(og_weights)

        return max(set(ensemble_actions), key=ensemble_actions.count)

    def prune_weights(self, weights, probability):
        w = copy.deepcopy(weights)
        for layer in w.keys():
            for weight in np.nditer(w[layer], op_flags=['readwrite']):
                if random() < probability:
                    weight[...] = 0
        return w
