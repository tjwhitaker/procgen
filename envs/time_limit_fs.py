import gym
import inspect
import numpy as np
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from collections import deque
from ray.tune import registry


class TimeLimit(gym.Wrapper):
    def __init__(self, env):
        super(TimeLimit, self).__init__(env)

        self.env = env
        self.episode_step = 0

    def reset(self, **kwargs):
        self.episode_step = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        # if self.env.env_name == 'coinrun' and self.episode_step > 300:
        #     state, reward, done, info = self.env.step(-1)
        # elif self.env.env_name == 'miner' and self.episode_step > 300:
        #     state, reward, done, info = self.env.step(-1)
        # elif self.env.env_name == 'bigfish' and self.episode_step > 850:
        #     state, reward, done, info = self.env.step(-1)
        # else:
        state, reward, done, info = self.env.step(action)

        if hasattr(self.env, 'rollout') and not self.env.rollout:
            if done and reward > 0:
                self.env.step(-1)
                done = False

        self.episode_step += 1

        return state, reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
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


registry.register_env(
    "time_limit_fs",
    lambda config: FrameStack(TimeLimit(ProcgenEnvWrapper(config)), 4),
)
