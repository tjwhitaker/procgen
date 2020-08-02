import numpy as np
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy


def my_postprocess_ppo_gae(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch["dones"][-1]

    # rmin = np.amin(sample_batch['rewards'])
    # rmax = np.amax(sample_batch['rewards'])

    # if (rmax - rmin) != 0:
    #     for i in range(len(sample_batch['rewards'])):
    #         sample_batch['rewards'][i] = (
    #             sample_batch['rewards'][i] - rmin) / (rmax - rmin)

    # sample_batch["rewards"] += self.rnd.calculate_bonus(sample_batch["obs"])

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


PseudoEnsemblePolicy = PPOTorchPolicy.with_updates(
    postprocess_fn=my_postprocess_ppo_gae,
)