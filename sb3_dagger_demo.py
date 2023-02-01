import tempfile
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

rng = np.random.default_rng(0)
env = gym.make("CartPole-v1")
expert = PPO(policy=MlpPolicy, env=env)
expert.learn(1000)
venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
)
with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng,
    )
    dagger_trainer.train(2000)

reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
print("Reward:", reward)
