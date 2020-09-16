import numpy as np
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin, ValueNetworkMixin

PseudoEnsemblePolicy = PPOTorchPolicy.with_updates(
    mixins=[LearningRateSchedule, EntropyCoeffSchedule,
            KLCoeffMixin, ValueNetworkMixin]
)
