import gym
import inspect
import numpy as np
from copy import copy
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from collections import deque
from ray.tune import registry


class ReduceActions(gym.Wrapper):
    def __init__(self, env):
        super(ReduceActions, self).__init__(env)

        # Remove diagonal directions?
        self.action_map = [1, 3, 5, 7, 9, 10, 11, 12, 13, 14]
        self.action_space = gym.spaces.Discrete(10)

    def step(self, action):
        return self.env.step(self.action_map[action])


# class ActionRewardHistory(gym.Wrapper):
#     def __init__(self, env):
#         super(ActionRewardHistory, self).__init__(env)
#         self.action_history = []
#         self.reward_history = []

#     def reset(self):
#         self.action_history = []
#         self.reward_history = []

#     def step(self, action):
#         state, reward, done, info = self.env.step(action)

#         self.action_history.append(action)
#         self.reward_history.append(reward)

#         return state, reward, done, info


class ContinuousLife(gym.Wrapper):
    def __init__(self, env, rollout):
        super(ContinuousLife, self).__init__(env)
        self.rollout = rollout

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if not self.rollout:
            if done and reward > 0:
                self.env.reset()
                done = False

        return state, reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class FrameSkip(gym.Wrapper):
    def __init__(self, env, n):
        super(FrameSkip, self).__init__(env)
        self.n = n

    def step(self, action):
        total_reward = 0

        for _ in range(self.n):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return state, total_reward, done, info


def create_env(config):
    config = copy(config)
    rollout = config.pop("rollout")
    env = ProcgenEnvWrapper(config)
    # env = ReduceActions(env)
    # env = ContinuousLife(env, rollout)
    env = FrameStack(env, 3)
    # env = FrameSkip(env, 1)
    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
