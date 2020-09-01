import numpy as np
import math
from copy import deepcopy
from random import random
from .pseudo_ensemble_trainer import PseudoEnsembleTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

import ray


def prune_weights(weights, probability):
    for layer in weights.keys():
        if "logits_fc.weight" in layer:
            for w in np.nditer(weights[layer], op_flags=['readwrite']):
                if random() < probability:
                    w[...] = 0
    return weights


def add_gaussian_noise(weights, scale):
    for layer in weights.keys():
        if "logits_fc.weight" in layer:
            for w in np.nditer(weights[layer], op_flags=['readwrite']):
                w[...] += np.random.normal(0, scale)
    return weights


def calculate_entropy(values):
    x = values - max(values)
    softmax = np.exp(x) / np.sum(np.exp(x))
    entropy = 0

    for p in softmax:
        entropy -= p * math.log(p, len(softmax))

    return entropy


class PseudoEnsembleAgent(PseudoEnsembleTrainer):
    def __init__(self, config=None, env=None, logger_creator=None):
        super().__init__(config, env, logger_creator)
        self.ensemble_weights = []
        self.original_weights = []

    # Should only be called during rollouts, not training
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

        self.global_vars["timestep"] += 1

        # Run through full network
        action, _, info = self.get_policy(policy_id).compute_single_action(
            filtered_obs,
            state,
            prev_action,
            prev_reward,
            info,
            clip_actions=self.config["clip_actions"],
            explore=True,
            timestep=self.global_vars["timestep"])

        # If entropy is high, run through ensemble
        if calculate_entropy(info['action_dist_inputs']) > 0.99:
            ensemble_actions = [action]

            for weights in self.ensemble_weights:
                self.get_policy().set_weights(weights)
                result = self.get_policy(policy_id).compute_single_action(
                    filtered_obs,
                    state,
                    prev_action,
                    prev_reward,
                    info,
                    clip_actions=self.config["clip_actions"],
                    explore=True,
                    timestep=self.global_vars["timestep"])
                ensemble_actions.append(result[0])

            self.get_policy().set_weights(self.original_weights)

            return max(set(ensemble_actions), key=ensemble_actions.count)
        else:
            return action

    def restore(self, checkpoint_path):
        super().restore(checkpoint_path)
        self.original_weights = self.get_policy().get_weights()
        self.create_ensemble()

    def create_ensemble(self):
        self.ensemble_weights = []
        alpha = 0.05

        for _ in range(4):
            w = deepcopy(self.original_weights)
            new_weights = prune_weights(w, alpha)
            # new_weights = add_gaussian_noise(w, alpha)
            self.ensemble_weights.append(new_weights)
