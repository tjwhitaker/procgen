import gym
import numpy as np
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from gym.wrappers import FrameStack
from ray.tune import registry


class NormalizeReward(gym.Wrapper):
    """
    Normalizes the reward space according to provided procgen constants.
    """

    def __init__(self, env):
        super(NormalizeReward, self).__init__(env)

        self.env_name = env.env_name
        self.constants = {
            'coinrun': [5., 10.],
            'starpilot': [2.5, 64.],
            'caveflyer': [3.5, 12.],
            'dodgeball': [1.5, 19.],
            'fruitbot': [-1.5, 32.4],
            'chaser': [.5, 13.],
            'miner': [1.5, 13.],
            'jumper': [3., 10.],
            'leaper': [3., 10.],
            'maze': [5., 10.],
            'bigfish': [1., 40.],
            'heist': [3.5, 10.],
            'climber': [2., 12.6],
            'plunder': [4.5, 30.],
            'ninja': [3.5, 10.],
            'bossfight': [.5, 13.],
            'caterpillar': [8.25, 24.]
        }

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, self.normalize(reward), done, info

    def normalize(self, value):
        r_range = self.constants[self.env_name]

        return ((value - 0) / (r_range[1] - 0))


registry.register_env(
    "normalize_reward_fs",
    lambda config: NormalizeReward(FrameStack(ProcgenEnvWrapper(config), 2)),
)
