from gym import Wrapper
from gym.wrappers import LazyFrames
from gym.spaces import Box
from collections import deque
import numpy as np
from ray.tune import registry
from envs.procgen_env_wrapper import ProcgenEnvWrapper


class FlowStack(Wrapper):
    def __init__(self, env):
        super(FlowStack, self).__init__(env)
        self.frames = deque(maxlen=2)

        low = np.repeat(
            self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype)

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(
            self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation()


registry.register_env(
    "flowstack_procgen_env",
    lambda config: FlowStack(ProcgenEnvWrapper(config)),
)
