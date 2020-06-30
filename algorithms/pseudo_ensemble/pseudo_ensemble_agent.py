import numpy as np
from copy import deepcopy
from random import random
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

import ray


class PseudoEnsembleAgent(PPOTrainer):
    def __init__(self, config=None, env=None, logger_creator=None):
        if "test_flag" in config:
            self.is_testing = True
            config.pop("test_flag")
        else:
            self.is_testing = False

        super().__init__(config, env, logger_creator)

        self.og_weights = []
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
        if self.is_testing:
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
        else:
            return super().compute_action(observation,
                                          state,
                                          prev_action,
                                          prev_reward,
                                          info,
                                          policy_id,
                                          full_fetch,
                                          explore)

    def restore(self, checkpoint_path):
        super().restore(checkpoint_path)

        if self.is_testing:
            self.ensemble_weights = []
            self.og_weights = self.get_policy().get_weights()

            for i in range(8):
                new_weights = self.prune_weights(self.og_weights, 0.1)
                self.ensemble_weights.append(new_weights)

    def prune_weights(self, weights, probability):
        w = deepcopy(weights)
        for layer in w.keys():
            if "hidden/kernel" in layer:
                for weight in np.nditer(w[layer], op_flags=['readwrite']):
                    if random() < probability:
                        weight[...] = 0
        return w

    def update_env():
        # destroy env
        # update config to hard mode
        # create new env
        pass
