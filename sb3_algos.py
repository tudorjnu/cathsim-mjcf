from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL

ALGOS = {
    'airl': AIRL,
    'gail': GAIL,
    'ppo': PPO,
    'sac': SAC,
    'recurrent_ppo': RecurrentPPO,
}
