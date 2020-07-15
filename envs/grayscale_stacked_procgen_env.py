from gym.wrappers import FrameStack, GrayScaleObservation
from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

registry.register_env(
    "grayscale_stacked_procgen_env",
    lambda config: FrameStack(
        GrayScaleObservation(ProcgenEnvWrapper(config)), 3),
)
