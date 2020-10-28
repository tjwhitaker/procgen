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

#######################################################
#######################################################
#######################################################
#######################################################
#######################################################

# Code from https://github.com/MadryLab/implementation-matters
# https://arxiv.org/abs/2005.12729


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''

    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass


class RewardFilter:
    """
    "Incorrect" reward normalization [copied from OAI code]
    Incorrect in the sense that we 
    1. update return
    2. divide reward by std(return) *without* subtracting and adding back mean
    """

    def __init__(self, prev_filter, shape, gamma):
        assert shape is not None
        self.gamma = gamma
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        return x

    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()


class NormalizeReward(gym.Wrapper):
    def __init__(self, env, rollout):
        super(NormalizeReward, self).__init__(env)
        self.rollout = rollout
        self.reward_filter = Identity()
        self.reward_filter = RewardFilter(
            self.reward_filter, shape=(), gamma=0.99)
        self.total_true_reward = 0.0

    def reset(self):
        start_state = self.env.reset()
        self.total_true_reward = 0.0
        self.reward_filter.reset()

        return start_state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.total_true_reward += reward

        if not self.rollout:
            reward = self.reward_filter(reward)

        return state, reward, done, info

#######################################################
#######################################################
#######################################################
#######################################################
#######################################################


def create_env(config):
    config = copy(config)
    rollout = config.pop("rollout")

    env = ProcgenEnvWrapper(config)
    env = DiffStack(env, 2)
    # env = FrameStack(env, 4)
    # env = ShapeReward(env, rollout)
    env = NormalizeReward(env, rollout)

    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
