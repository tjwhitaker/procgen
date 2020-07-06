import numpy as np
from copy import deepcopy
from random import random
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

import ray


def prune_weights(weights, probability):
    w = deepcopy(weights)

    for layer in weights.keys():
        if ("pi" not in layer) and ("vf" not in layer):
            for weight in np.nditer(w[layer], op_flags=['readwrite']):
                if random() < probability:
                    weight[...] = 0

    return weights


def add_gaussian_noise(weights, scale):
    w = deepcopy(weights)

    for layer in weights.keys():
        if ("pi" not in layer) and ("vf" not in layer):
            for weight in np.nditer(w[layer], op_flags=['readwrite']):
                weight[...] += np.random.normal(0, scale)

    return weights


class PseudoEnsembleAgent(PPOTrainer):
    def __init__(self, config=None, env=None, logger_creator=None):
        super().__init__(config, env, logger_creator)
        self.ensemble_weights = []

    def compute_action(self,
                       observation,
                       state=None,
                       prev_action=None,
                       prev_reward=None,
                       info=None,
                       policy_id=DEFAULT_POLICY_ID,
                       full_fetch=False,
                       explore=None):
        if state is None:
            state = []
        preprocessed = self.workers.local_worker().preprocessors[
            policy_id].transform(observation)
        filtered_obs = self.workers.local_worker().filters[policy_id](
            preprocessed, update=False)

        # Figure out the current (sample) time step and pass it into Policy.
        self.global_vars["timestep"] += 1

        ensemble_actions = []

        for weights in self.ensemble_weights:
            self.get_policy().set_weights(weights)
            result = self.get_policy(policy_id).compute_single_action(
                filtered_obs,
                state,
                prev_action,
                prev_reward,
                info,
                clip_actions=self.config["clip_actions"],
                explore=explore,
                timestep=self.global_vars["timestep"])
            ensemble_actions.append(result[0])

        return max(set(ensemble_actions), key=ensemble_actions.count)

    def restore(self, checkpoint_path):
        super().restore(checkpoint_path)
        self.create_ensemble()

    def create_ensemble(self):
        self.ensemble_weights = []
        weights = self.get_policy().get_weights()

        for i in range(8):
            # new_weights = prune_weights(weights, 0.05)
            new_weights = add_gaussian_noise(weights, 1.0)

            self.ensemble_weights.append(new_weights)
