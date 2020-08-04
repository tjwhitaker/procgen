import gym
import numpy as np
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from gym.wrappers import FrameStack
from ray.tune import registry


class CuriosityBonus(gym.Wrapper):
    """
    Grant a curiosity bonus based on a perceptual hash.
    """

    def __init__(self, env):
        super(CuriosityBonus, self).__init__(env)

        self.episode_reward = 0
        self.episode_step = 0
        self.state_history = {}

    def reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_step = 0
        self.state_history = {}

        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # Time Bonus
        if reward != 0:
            if self.episode_step < 40:
                reward += 1
            elif self.episode_step < 80:
                reward += 0.5
            elif self.episode_step < 160:
                reward += 0.1

        # Curiosity Bonus
        if self.env.env_name != "coinrun":
            bucket = int(self.episode_reward / 10)
            key = str(bucket)

            for i in range(3):
                channel_sum = np.sum(state[:, :, i])
                key += "," + str(int(channel_sum / 33333))

            if key not in self.state_history:
                self.state_history[key] = True
                reward += 0.1

        self.episode_reward += reward
        self.episode_step += 1

        return state, reward, done, info


registry.register_env(
    "curiosity_bonus_fs",
    lambda config: FrameStack(CuriosityBonus(ProcgenEnvWrapper(config)), 2),
)
