import gym
import numpy as np
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from gym.wrappers import FrameStack
from ray.tune import registry


class TimeBonus(gym.Wrapper):
    def __init__(self, env):
        super(TimeBonus, self).__init__(env)
        self.ts = 0

    def step(self, ac):
        self.ts += 1
        observation, reward, done, info = self.env.step(ac)
        if self.ts < 256:
            reward = reward + (reward * (256 - self.ts) * 0.001)
        return observation, reward, done, info

    def reset(self):
        self.ts = 0
        return self.env.reset()


registry.register_env(
    "time_bonus_fs",
    lambda config: FrameStack(TimeBonus(ProcgenEnvWrapper(config)), 2),
)
