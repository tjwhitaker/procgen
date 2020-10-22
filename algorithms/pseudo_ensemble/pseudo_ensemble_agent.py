import numpy as np
import math
from copy import deepcopy
from random import random
from .pseudo_ensemble_trainer import PseudoEnsembleTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

import ray


class PseudoEnsembleAgent(PseudoEnsembleTrainer):
    def __init__(self, config=None, env=None, logger_creator=None):
        super().__init__(config, env, logger_creator)

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

        preprocessed = self.workers.local_worker().preprocessors[
            policy_id].transform(observation)
        filtered_obs = self.workers.local_worker().filters[policy_id](
            preprocessed, update=False)

        model = self.get_policy(policy_id).model
        action = model.ensemble_forward(filtered_obs, 0)

        return action

    def restore(self, checkpoint_path):
        super().restore(checkpoint_path)
        self.get_policy(DEFAULT_POLICY_ID).model.create_ensemble(0)
