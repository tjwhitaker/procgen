import numpy as np
import copy
from random import random
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

import ray

from multiprocessing import Pool


def predict(data):
    print(data)
    return 1


class PseudoEnsembleAgent(PPOTrainer):
    def __init__(self, config=None, env=None, logger_creator=None):
        super().__init__(config, env, logger_creator)

        self.ensemble_weights = []
        self.og_weights = self.get_policy().get_weights()

        for i in range(4):
            new_weights = self.prune_weights(self.og_weights, 0.1)
            self.ensemble_weights.append(new_weights)

        # self.ensemble = []

        # for i in range(4):
        #     self.ensemble.append(
        #         self.get_policy().model.base_model.get_config())

    def compute_action(self,
                       observation,
                       state=None,
                       prev_action=None,
                       prev_reward=None,
                       info=None,
                       policy_id=DEFAULT_POLICY_ID,
                       full_fetch=False,
                       explore=None):
        ensemble_actions = []

        for weights in self.ensemble_weights:
            self.get_policy().set_weights(weights)
            ensemble_actions.append(super().compute_action(observation, state, prev_action,
                                                           prev_reward, info, policy_id, full_fetch, explore))

        return 1
        # return max(set(ensemble_actions), key=ensemble_actions.count)

    def prune_weights(self, weights, probability):
        w = copy.deepcopy(weights)
        for layer in w.keys():
            for weight in np.nditer(w[layer], op_flags=['readwrite']):
                if random() < probability:
                    weight[...] = 0
        return w
