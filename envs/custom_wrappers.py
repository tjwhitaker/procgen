import gym
import inspect
import numpy as np
from copy import copy
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from collections import deque
from ray.tune import registry


class TimeLimit(gym.Wrapper):
    def __init__(self, env, rollout):
        super(TimeLimit, self).__init__(env)
        self.episode_step = 0
        self.rollout = rollout

    def reset(self):
        self.episode_step = 0
        return self.env.reset()

    def step(self, action):
        self.episode_step += 1

        # Only implement time limit for training
        if not self.rollout:
            if self.env.env_name == 'coinrun' and self.episode_step > 350:
                state, reward, done, info = self.env.step(-1)
            elif self.env.env_name == 'miner' and self.episode_step > 350:
                state, reward, done, info = self.env.step(-1)
            elif self.env.env_name == 'bigfish' and self.episode_step > 850:
                state, reward, done, info = self.env.step(-1)
            else:
                state, reward, done, info = self.env.step(action)
        else:
            state, reward, done, info = self.env.step(action)

        return state, reward, done, info


class ContinuousLife(gym.Wrapper):
    def __init__(self, env, rollout):
        super(ContinuousLife, self).__init__(env)
        self.rollout = rollout

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # When we complete a level:
        # Load the next level without ending the episode
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
        done = False
        total_reward = 0

        if self.env.env_name == "bigfish" or self.env.env_name == "coinrun":
            for _ in range(self.n):
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break
        else:
            state, total_reward, done, info = self.env.step(action)

        return state, total_reward, done, info


def create_env(config):
    config = copy(config)
    rollout = config.pop("rollout")
    env = ProcgenEnvWrapper(config)
    env = TimeLimit(env, rollout)
    env = ContinuousLife(env, rollout)
    env = FrameStack(env, 4)
    env = FrameSkip(env, 2)
    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
