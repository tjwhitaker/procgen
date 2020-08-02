import gym
import numpy as np
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from gym.wrappers import FrameStack
from ray.tune import registry


class ReduceActionSpace(gym.Wrapper):
    """
    Normalizes the reward space according to provided procgen constants.
    """

    def __init__(self, env):
        super(ReduceActionSpace, self).__init__(env)
        self.action_space = gym.spaces.Discrete(9)


registry.register_env(
    "reduce_action_space",
    lambda config: FrameStack(ReduceActionSpace(ProcgenEnvWrapper(config)), 2),
)
