from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.mbmpo.mbmpo import MBMPOConfig
from ray.rllib.algorithms.dreamer.dreamer import DreamerConfig

CONFIGS = {
    "PPO": PPOConfig,
    "SAC": SACConfig,
    "MBMPO": MBMPOConfig,
    "Dreamer": DreamerConfig,
}
