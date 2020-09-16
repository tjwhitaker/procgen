import numpy as np
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin, ValueNetworkMixin

from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy


def my_postprocess_ppo_gae(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch["dones"][-1]

    # Log Scale Rewards
    # for i in range(len(sample_batch['rewards'])):
    #     if sample_batch['rewards'][i] >= 0:
    #         sample_batch['rewards'][i] = np.log10(sample_batch['rewards'][i]+1)

    # Standardize
    # Does not work. Mean shift causes problem
    # sample_batch['rewards'] = (sample_batch['rewards'] - np.mean(
    #     sample_batch['rewards'])) / (np.std(sample_batch['rewards']) + 1e-16)

    # Normalize
    # rmin = np.amin(sample_batch['rewards'])
    # rmax = np.amax(sample_batch['rewards'])

    # if ((rmax - rmin) != 0) and (rmax >= 1):
    #     for i in range(len(sample_batch['rewards'])):
    #         sample_batch['rewards'][i] = (
    #             sample_batch['rewards'][i] - rmin) / (rmax - rmin)

    # Baseline Reduction
    # total = np.sum(sample_batch['rewards'])

    # for i in range(len(sample_batch['rewards'])):
    #     if total != 0:
    #         sample_batch['rewards'][i] = sample_batch['rewards'][i] - \
    #             (1/total * sample_batch['rewards'][i])

    # Clamp Rewards
    # for i in range(len(sample_batch['rewards'])):
    #     if sample_batch['rewards'][i] > 1:
    #         sample_batch['rewards'][i] = 1
    #     elif sample_batch['rewards'][i] < -1:
    #         sample_batch['rewards'][i] = -1

    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch


# class PseudoEnsemblePolicy(PPOTorchPolicy):
#     def __init__(self):
#         pass

    # def compute_single_action():

    # PseudoEnsemblePolicy = PPOTorchPolicy.with_updates(
    #     postprocess_fn=my_postprocess_ppo_gae,
    #     mixins=[LearningRateSchedule, EntropyCoeffSchedule,
    #             KLCoeffMixin, ValueNetworkMixin]
    # )
