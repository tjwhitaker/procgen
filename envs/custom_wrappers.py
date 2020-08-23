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
        self.action_map = []
        self.test_action_space()

    def step(self, action):
        return self.env.step(self.action_map[action])

    # Environment Independent Action Reduction
    def test_action_space(self):
        # Initial reset needed for monitor wrapper
        self.env.reset()

        eliminate_actions = []
        base_state = self.unwrapped.env.env.callmethod("get_state")

        ######################
        # Test Special Actions
        ######################
        astates = []
        for _ in range(10):
            a, _, _, _ = self.env.step(4)
            astates.append(a)

        self.unwrapped.env.env.callmethod("set_state", base_state)

        for action in [9, 10, 11, 12, 13, 14]:
            bstates = []
            for _ in range(10):
                b, _, _, _ = self.env.step(action)
                bstates.append(b)

            state_checks = []
            for i in range(10):
                eql = astates[i] == bstates[i]
                state_checks.append(eql.all())

            if all(state_checks):
                eliminate_actions.append(action)

            self.unwrapped.env.env.callmethod("set_state", base_state)

        ######################################
        # Test Diagonal == Horizontal Movement
        ######################################
        for _ in range(5):
            la, _, _, _ = self.env.step(1)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(5):
            lb, _, _, _ = self.env.step(0)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(5):
            lc, _, _, _ = self.env.step(2)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(5):
            ra, _, _, _ = self.env.step(7)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(5):
            rb, _, _, _ = self.env.step(6)
        self.unwrapped.env.env.callmethod("set_state", base_state)

        for _ in range(5):
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
    # env = FrameSkip(env, 1)
    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
