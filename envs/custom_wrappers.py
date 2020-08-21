import gym
import inspect
import numpy as np
from copy import copy
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from collections import deque
from ray.tune import registry


# class DeliberatePractice(gym.Wrapper):
#     def __init__(self, env):
#         super(DeliberatePractice, self).__init__(env)
#         self.env_state = []

#     def reset(self):
#         self.env_state = self.env.env.env.callmethod("get_state")
#         print(states)


class ReduceActions(gym.Wrapper):
    def __init__(self, env):
        super(ReduceActions, self).__init__(env)
        self.action_space = self.test_action_space()

    def test_action_space(self):
        truth = []
        base_state = self.env.env.env.callmethod("get_state")

        # Test Special Actions
        for _ in range(5):
            a, _, _, _ = self.env.step(4)

        self.env.env.env.callmethod("set_state", base_state)

        for action in [9, 10, 11, 12, 13, 14]:
            for _ in range(5):
                b, _, _, _ = self.env.step(action)
            test_state = a == b
            truth.append(test_state.all())
            self.env.env.env.callmethod("set_state", base_state)

        self.env.step(-1)

        if all(truth):
            return gym.spaces.Discrete(9)
        else:
            return gym.spaces.Discrete(15)


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
    env = ReduceActions(env)
    # env = ContinuousLife(env, rollout)
    env = FrameStack(env, 4)
    # env = FrameSkip(env, 2)
    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
