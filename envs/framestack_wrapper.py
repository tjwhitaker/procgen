from gym.wrappers import FrameStack
from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

registry.register_env(
    "stacked_procgen_env",
    lambda config: FrameStack(ProcgenEnvWrapper(config), 2),
)
