import numpy as np
import copy
from random import random

from itertools import repeat
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

from multiprocessing import Pool

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from ray.util.iter import from_items

from tensorflow.keras.models import clone_model


def predict(data):
    print(data)
    return 1


class PseudoEnsembleAgent(PPOTrainer):
    def __init__(self, config=None, env=None, logger_creator=None):
        config["num_workers"] = 0
        super().__init__(config, env, logger_creator)

        self.ensemble_weights = []
        self.og_weights = self.get_policy().get_weights()

        for i in range(4):
            new_weights = self.prune_weights(self.og_weights, 0.1)
            self.ensemble_weights.append(new_weights)

        # self.ensemble = []

        # for i in range(4):
        #     self.ensemble.append(clone_model(
        #         self.get_policy().model.base_model))

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
        # pool = Pool(8)
        # ensemble_actions = pool.starmap(predict, zip([self] * 8, self.ensemble))

        # ensemble_args = zip(self.ensemble_weights)
        # ensemble_actions = Parallel(n_jobs=4)(
        #     delayed(unwrap_self)(w) for w in self.ensemble)

        # it = from_items(self.ensemble, num_shards=8).for_each(
        #     lambda x: self.super_compute_action(x, observation, state, prev_action, prev_reward, info, policy_id, full_fetch, explore)).gather_async()
        # ensemble_actions = it.take(8)

        # print(ensemble_actions)

        # nn = self.get_policy().model
        # print(nn)

        ensemble_actions = []

        for weights in self.ensemble_weights:
            self.get_policy().set_weights(weights)
            ensemble_actions.append(super().compute_action(observation, state, prev_action,
                                                           prev_reward, info, policy_id, full_fetch, explore))

        # # Reset Weights
        # self.get_policy().set_weights(self.og_weights)
        # print(ensemble_actions)

        return max(set(ensemble_actions), key=ensemble_actions.count)

    def prune_weights(self, weights, probability):
        w = copy.deepcopy(weights)
        for layer in w.keys():
            for weight in np.nditer(w[layer], op_flags=['readwrite']):
                if random() < probability:
                    weight[...] = 0
        return w
