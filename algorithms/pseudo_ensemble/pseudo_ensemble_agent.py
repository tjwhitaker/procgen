import numpy as np
from copy import deepcopy
from random import random
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import merge_dicts

import ray
import cma


def prune_weights(weights, probability):
    for layer in weights.keys():
        if "hidden_fc.weight" in layer:
            for w in np.nditer(weights[layer], op_flags=['readwrite']):
                if random() < probability:
                    w[...] = 0

    return weights


def add_gaussian_noise(weights, scale):
    for layer in weights.keys():
        if "hidden_fc.weight" in layer:
            for w in np.nditer(weights[layer], op_flags=['readwrite']):
                w[...] += np.random.normal(0, scale)

    return weights


def calculate_confidence(values):
    x = values - max(values)
    softmax = np.exp(x) / np.sum(np.exp(x))
    return max(softmax)


class PseudoEnsembleAgent(PPOTrainer):
    def __init__(self, config=None, env=None, logger_creator=None):
        super().__init__(config, env, logger_creator)
        self.ensemble_weights = []
        self.original_weights = []

        # extra_config = deepcopy(self.config["evaluation_config"])
        # extra_config.update({
        #     "batch_mode": "complete_episodes",
        #     "rollout_fragment_length": 1,
        #     "in_evaluation": True,
        # })

        # self.config["evaluation_num_episodes"] = 25
        # self.config["evaluation_num_workers"] = 0

        # self.evaluation_workers = self._make_workers(
        #     self.env_creator,
        #     self._policy,
        #     merge_dicts(self.config, extra_config),
        #     num_workers=0)

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
            explore=explore,
            timestep=self.global_vars["timestep"])

        # If confidence is low, run through ensemble
        if calculate_confidence(info['action_dist_inputs']) < 0.65:
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

        # self.evolve()

        for i in range(8):
            w = deepcopy(self.original_weights)
            # new_weights = prune_weights(w, 0.1)
            new_weights = add_gaussian_noise(
                deepcopy(self.original_weights), 0.1)

            self.ensemble_weights.append(new_weights)

    # def evolve(self):
    #     vector = deepcopy(self.original_weights['logits_fc.weight'])
    #     fv = vector.flatten()
    #     es = cma.CMAEvolutionStrategy(fv, 0.05)
    #     es.optimize(self.evolve_eval, iterations=10, verb_disp=1)
    #     pass

    # def evolve_eval(self, x):
    #     new_weights = deepcopy(self.original_weights)
    #     new_weights['logits_fc.weight'] = x.reshape(15, 256)
    #     self.get_policy().set_weights(new_weights)
    #     metrics = self._evaluate()
    #     print(metrics['evaluation']['episode_reward_mean'])
    #     return metrics['evaluation']['episode_reward_max'] - metrics['evaluation']['episode_reward_mean']
