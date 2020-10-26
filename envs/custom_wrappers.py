import gym
import inspect
import numpy as np
from copy import copy
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from collections import deque
from ray.tune import registry
from random import random


class FrameStack(gym.Wrapper):
    def __init__(self, env, n):
        super(FrameStack, self).__init__(env)
        self.n = n
        self.frames = deque([], maxlen=n)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.n):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n
        return np.concatenate(self.frames, axis=2)


class DiffStack(gym.Wrapper):
    def __init__(self, env, n):
        super(DiffStack, self).__init__(env)
        self.n = n
        self.frames = deque([], maxlen=n)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.n):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n
        frames = [self.frames[1], abs(self.frames[1] - self.frames[0])]
        return np.concatenate(frames, axis=2)


# Log scale rewards
class ShapeReward(gym.Wrapper):
    def __init__(self, env, rollout):
        super(ShapeReward, self).__init__(env)
        self.rollout = rollout

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if not self.rollout:
            if reward > 1:
                reward = np.log(reward+2)

        return state, reward, done, info


def create_env(config):
    config = copy(config)

    env = ProcgenEnvWrapper(config)
    env = DiffStack(env, 2)
    # env = FrameStack(env, 4)
    # env = ShapeReward(env, rollout)

    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
