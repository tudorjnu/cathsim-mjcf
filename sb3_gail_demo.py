from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

from imitation.data import rollout
from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms.adversarial.gail import GAIL
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper

import numpy as np
import gym
from pathlib import Path
import seals

expert_path = Path.cwd() / 'rl' / 'sb3' / 'test'
log_path = expert_path / 'logs'
model_path = expert_path / 'checkpoints'

for path in [log_path, model_path]:
    path.mkdir(parents=True, exist_ok=True)

env = gym.make("seals/CartPole-v0")
expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)
expert.learn(1000)  # Note: set to 100000 to train a proficient expert


rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    make_vec_env(
        "seals/CartPole-v0",
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)


venv = make_vec_env("seals/CartPole-v0", n_envs=8, rng=rng)

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
)

reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm
)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

checkpoint_callback = CheckpointCallback(
    save_freq=400,
    save_path=expert_path.as_posix(),
    name_prefix="test",
    save_replay_buffer=True,
    save_vecnormalize=True,
    verbose=1,
)

rewards, lengths = evaluate_policy(
    learner, venv, 10, return_episode_rewards=True)
print("Before training:")
print(f"\tMean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
print(f"\tMean length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")

gail_trainer.train(300000)
rewards, lengths = evaluate_policy(
    learner, venv, 10, return_episode_rewards=True)

print("After training")
print(f"\tMean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
print(f"\tMean length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")
learner.save(model_path)
