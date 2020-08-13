import gym
import inspect
import numpy as np
from copy import copy
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from collections import deque
from ray.tune import registry

class EpisodicLife(gym.Wrapper):
    def __init__(self, env, rollout):
        super(EpisodicLife, self).__init__(env)
        self.rollout = rollout

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if not self.rollout:
            if done and reward > 0:
                self.env.reset()
                done = False

        return state, reward, done, info

class TimeLimit(gym.Wrapper):
    def __init__(self, env):
        super(TimeLimit, self).__init__(env)
        self.episode_step = 0

    def reset(self, **kwargs):
        self.episode_step = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        self.episode_step += 1

        if self.env.env_name == 'coinrun' and self.episode_step > 500:
            state, reward, done, info = self.env.step(-1)
        elif self.env.env_name == 'miner' and self.episode_step > 500:
            state, reward, done, info = self.env.step(-1)
        elif self.env.env_name == 'bigfish' and self.episode_step > 1000:
            state, reward, done, info = self.env.step(-1)
        else:
            state, reward, done, info = self.env.step(action)

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

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        done = False
        total_reward = 0

        if self.env.env_name == "bigfish" or self.env.env_name == "coinrun":
            for _ in range(self.n):
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                if done: break
        else:
            state, total_reward, done, info = self.env.step(action)

        return state, total_reward, done, info

def create_env(config):
    config = copy(config)
    rollout = config.pop("rollout")
    env = ProcgenEnvWrapper(config)
    # env = EpisodicLife(env, rollout)
    env = TimeLimit(env)
    env = FrameStack(env, 4)
    env = FrameSkip(env, 4)
    return env

registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
