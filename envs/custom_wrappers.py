import gym
import inspect
import numpy as np
from copy import copy
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from collections import deque
from ray.tune import registry
from random import random


class ReduceActions(gym.Wrapper):
    def __init__(self, env):
        super(ReduceActions, self).__init__(env)
        self.action_map = []
        self.reduce_action_space()
        print(self.action_space)

    def step(self, action):
        if action >= len(self.action_map):
            return self.env.step(action)
        else:
            return self.env.step(self.action_map[action])

    # Environment Independent Action Reduction
    def reduce_action_space(self):
        # Initial reset needed for monitor wrapper
        self.env.reset()

        eliminate_actions = []
        base_state = self.unwrapped.env.env.callmethod("get_state")

        # Test Special Actions
        astates = []
        for _ in range(10):
            a, _, done, _ = self.env.step(4)
            astates.append(a)
            if done:
                break

        self.unwrapped.env.env.callmethod("set_state", base_state)

        for action in [9, 10, 11, 12, 13, 14]:
            bstates = []
            for _ in range(10):
                b, _, done, _ = self.env.step(action)
                bstates.append(b)
                if done:
                    break

            state_checks = []

            if len(astates) == len(bstates):
                for i in range(len(astates)):
                    eql = astates[i] == bstates[i]
                    state_checks.append(eql.all())

                if all(state_checks):
                    eliminate_actions.append(action)

            self.unwrapped.env.env.callmethod("set_state", base_state)

        # Test Diagonal Movement
        # for _ in range(5):
        #     la, _, done, _ = self.env.step(1)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # for _ in range(5):
        #     lb, _, done, _ = self.env.step(0)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # for _ in range(5):
        #     lc, _, done, _ = self.env.step(2)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # for _ in range(5):
        #     ra, _, done, _ = self.env.step(7)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # for _ in range(5):
        #     rb, _, done, _ = self.env.step(6)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # for _ in range(5):
        #     rc, _, done, _ = self.env.step(8)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # # State Comparisons
        # lld = la == lb
        # llu = la == lc
        # rrd = ra == rb
        # rru = ra == rc

        # # Enforce symmetry if we remove diagonals
        # if lld.all() and llu.all() and rrd.all() and rru.all():
        #     eliminate_actions.append(0)
        #     eliminate_actions.append(2)
        #     eliminate_actions.append(6)
        #     eliminate_actions.append(8)

        # Build our action map
        actions = set([*range(15)])
        eliminations = set(eliminate_actions)
        self.action_map = list(actions - eliminations)
        self.action_space = gym.spaces.Discrete(len(self.action_map))

        # Force reset the env to start training
        self.env.step(-1)


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


class NormalizeRewards(gym.Wrapper):
    def __init__(self, env, rollout, return_max, return_min, return_blind):
        super(NormalizeRewards, self).__init__(env)
        self.rollout = rollout
        self.max = return_max
        self.min = return_min
        self.blind = abs(return_blind)
        self.current_max = 1.0

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if reward > self.current_max:
            self.current_max = reward

        if not self.rollout:
            reward = (reward - self.min) / (self.current_max - self.min)

        return state, reward, done, info


def create_env(config):
    config = copy(config)
    # rollout = config.pop("rollout")
    # return_max = config.pop("return_max")
    # return_min = config.pop("return_min")
    # return_blind = config.pop("return_blind")
    env = ProcgenEnvWrapper(config)
    env = ReduceActions(env)
    # env = NormalizeRewards(env, rollout, return_max, return_min, return_blind)
    env = DiffStack(env, 2)
    # env = ContinuousLife(env, rollout, reward_max)
    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
