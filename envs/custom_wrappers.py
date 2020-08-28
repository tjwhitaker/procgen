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

        # Test Diagonal == Horizontal Movement
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
        self.episode_reward = 0

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

        self.episode_reward += reward

        if not self.rollout:
            # Need to know max reward to know if we've completed level
            # Previous solution used done && current step reward > 0
            # Errors when you get a reward and die in the same frame (bigfish)
            if done and (self.episode_reward >= self.reward_max[self.env.env_name]):
                self.episode_reward = 0
                done = False

        return state, reward, done, info


class DeliberatePractice(gym.Wrapper):
    def __init__(self, env, rollout):
        super(DeliberatePractice, self).__init__(env)
        self.rollout = rollout
        self.base_state = []
        self.base_obs = []
        self.episode_reward = 0

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

    def reset(self):
        self.episode_reward = 0
        self.base_obs = self.env.reset()
        self.base_state = self.unwrapped.env.env.callmethod("get_state")
        return self.base_obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.episode_reward += reward

        if not self.rollout:
            if done and (self.episode_reward < self.reward_max[self.env.env_name]):
                self.unwrapped.env.env.callmethod("set_state", self.base_state)
                return self.base_obs, reward, False, info

        return state, reward, done, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env, rollout):
        super(TimeLimit, self).__init__(env)
        self.steps = 0
        self.rollout = rollout

    def step(self, action):
        self.steps += 1

        if (not self.rollout) and (self.steps > 1000):
            self.steps = 0
            return self.env.step(-1)
        else:
            return self.env.step(action)


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


def create_env(config):
    config = copy(config)
    rollout = config.pop("rollout")
    env = ProcgenEnvWrapper(config)
    env = ReduceActions(env)
    # env = ContinuousLife(env, rollout)
    # env = DeliberatePractice(env, rollout)
    # env = TimeLimit(env, rollout)
    env = FrameStack(env, 4)
    return env


registry.register_env(
    "custom_wrapper", lambda config: create_env(config),
)
