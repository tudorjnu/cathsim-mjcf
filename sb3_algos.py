from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from imitation.algorithms.adversarial.airl import AIRL

ALGOS = {
    'airl': AIRL,
    'ppo': PPO,
    'sac': SAC,
    'recurrent_ppo': RecurrentPPO,
}
