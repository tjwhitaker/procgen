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

    def step(self, action):
        return self.env.step(self.action_map[action])

    # Environment Independent Action Reduction
    def reduce_action_space(self):
        # Initial reset needed for monitor wrapper
        self.env.reset()

        eliminate_actions = []
        base_state = self.unwrapped.env.env.callmethod("get_state")

        # Test Special Actions
        astates = []
        for _ in range(8):
            a, _, _, _ = self.env.step(4)
            astates.append(a)

        self.unwrapped.env.env.callmethod("set_state", base_state)

        for action in [9, 10, 11, 12, 13, 14]:
            bstates = []
            for _ in range(8):
                b, _, _, _ = self.env.step(action)
                bstates.append(b)

            state_checks = []
            for i in range(8):
                eql = astates[i] == bstates[i]
                state_checks.append(eql.all())

            if all(state_checks):
                eliminate_actions.append(action)

            self.unwrapped.env.env.callmethod("set_state", base_state)

        # Test Diagonal == Horizontal Movement
        for _ in range(4):
            la, _, _, _ = self.env.step(1)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(4):
            lb, _, _, _ = self.env.step(0)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(4):
            lc, _, _, _ = self.env.step(2)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(4):
            ra, _, _, _ = self.env.step(7)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(4):
            rb, _, _, _ = self.env.step(6)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(4):
            rc, _, _, _ = self.env.step(8)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        # State Comparisons
        lld = la == lb
        llu = la == lc
        rrd = ra == rb
        rru = ra == rc

        # Enforce symmetry if we remove diagonals
        if lld.all() and llu.all() and rrd.all() and rru.all():
            eliminate_actions.append(0)
            eliminate_actions.append(2)
            eliminate_actions.append(6)
            eliminate_actions.append(8)

        # Build our action map
        actions = set([*range(15)])
        eliminations = set(eliminate_actions)
        self.action_map = list(actions - eliminations)
        self.action_space = gym.spaces.Discrete(len(self.action_map))

        # Force reset the env to start training
        self.env.step(-1)


class ContinuousLife(gym.Wrapper):
    def __init__(self, env, rollout):
        super(ContinuousLife, self).__init__(env)
        self.rollout = rollout
        self.episode_reward = 0
        self.episode_step = 0

        # See https://discourse.aicrowd.com/t/getting-rmax-from-environment/3362
        self.reward_max = {
            'coinrun': 10,
            'starpilot': 64,
            'caveflyer': 12,
            'dodgeball': 19,
            'fruitbot': 32.4,
            'chaser': 13,
            'miner': 13,
            'jumper': 10,
            'leaper': 10,
            'maze': 10,
            'bigfish': 40,
            'heist': 10,
            'climber': 12.6,
            'plunder': 30,
            'ninja': 10,
            'bossfight': 13,
            'caterpillar': 24,
        }

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.episode_step += 1
        self.episode_reward += reward

        if not self.rollout and done:
            if self.episode_reward >= self.reward_max[self.env.env_name]:
                self.env.reset()
                done = False

            self.episode_step = 0
            self.episode_reward = 0

        return state, reward, done, info


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


class DiffFrameStack(gym.Wrapper):
    def __init__(self, env, n):
        super(DiffFrameStack, self).__init__(env)
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
        frames = [self.frames[-1],
                  abs(self.frames[-1] - self.frames[-2]),
                  abs(self.frames[-1] - self.frames[-3])]
        return np.concatenate(frames, axis=2)


def create_env(config):
    config = copy(config)
    rollout = config.pop("rollout")
    env = ProcgenEnvWrapper(config)
    env = ReduceActions(env)
    env = DiffFrameStack(env, 3)
    env = ContinuousLife(env, rollout)
    # env = ReloadLevels(env, rollout)
    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
