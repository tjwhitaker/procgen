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

        # movement_states = {}

        # Test Diagonal Movement
        # Left
        # for _ in range(5):
        #     s, _, done, _ = self.env.step(1)
        #     movement_states["l"].append(s)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # # Left Down
        # for _ in range(5):
        #     s, _, done, _ = self.env.step(0)
        #     movement_states["ld"].append(s)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # # Left Up
        # for _ in range(5):
        #     s, _, done, _ = self.env.step(2)
        #     movement_states["lu"].append(s)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # # Up
        # for _ in range(5):
        #     s, _, done, _ = self.env.step(5)
        #     movement_states["u"].append(s)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # # Right
        # for _ in range(5):
        #     s, _, done, _ = self.env.step(7)
        #     movement_states["r"].append(s)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # # Right Down
        # for _ in range(5):
        #     s, _, done, _ = self.env.step(6)
        #     movement_states["rd"].append(s)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # # Right Up
        # for _ in range(5):
        #     s, _, done, _ = self.env.step(8)
        #     movement_states["ru"].append(s)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # # Down
        # for _ in range(5):
        #     s, _, done, _ = self.env.step(3)
        #     movement_states["d"].append(s)
        #     if done:
        #         break
        # self.unwrapped.env.env.callmethod("set_state", base_state)

        # # Diagonal == horizontal?
        # lld = la == lb
        # llu = la == lc
        # rrd = ra == rb
        # rru = ra == rc

        # Diagonal == vertical?
        # ulu =
        # uru =
        # dld =
        # drd =

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


class ContinuousLife(gym.Wrapper):
    def __init__(self, env, rollout, reward_max):
        super(ContinuousLife, self).__init__(env)
        self.rollout = rollout
        self.episode_reward = 0
        self.episode_step = 0
        self.reward_max = reward_max

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.episode_step += 1
        self.episode_reward += reward

        if not self.rollout and done:
            if self.episode_reward >= self.reward_max:
                self.env.reset()
                done = False

            self.episode_step = 0
            self.episode_reward = 0

        return state, reward, done, info


# Log scale rewards
class ShapeReward(gym.Wrapper):
    def __init__(self, env, rollout):
        super(ShapeReward, self).__init__(env)
        self.rollout = rollout

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if (not self.rollout) and (reward > 1):
            reward = np.log(reward+1)

        return state, reward, done, info


def create_env(config):
    config = copy(config)
    rollout = config.pop("rollout")

    env = ProcgenEnvWrapper(config)
    env = ReduceActions(env)
    # env = DiffStack(env, 2)
    # env = ContinuousLife(env, rollout, return_max)
    env = FrameStack(env, 3)
    env = ShapeReward(env, rollout)

    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
