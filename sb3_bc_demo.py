import gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

expert_path = Path.cwd() / 'rl' / 'sb3' / 'test'
log_path = expert_path / 'logs' / 'bc'
model_path = expert_path / 'checkpoints'

rng = np.random.default_rng(0)
env = gym.make("CartPole-v1")
expert = PPO(policy=MlpPolicy, env=env)
expert.learn(1000)

rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)

transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=10)
reward, lengths = evaluate_policy(
    bc_trainer.policy, env, 10, return_episode_rewards=True)
print("Reward:", np.mean(reward))
print("Lengths:", np.mean(lengths))

bc_trainer.save_policy(model_path / 'bc')
