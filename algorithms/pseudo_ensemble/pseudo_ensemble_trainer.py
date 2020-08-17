from ray.rllib.agents.ppo import PPOTrainer
from .pseudo_ensemble_policy import PseudoEnsemblePolicy


def get_policy_class(config):
    return PseudoEnsemblePolicy


# PseudoEnsembleTrainer = PPOTrainer.with_updates(
#     default_policy=PseudoEnsemblePolicy,
#     get_policy_class=get_policy_class
# )

PseudoEnsembleTrainer = PPOTrainer
