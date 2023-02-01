import numpy as np
import gym
import seals
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

rng = np.random.default_rng(0)

env = gym.make("seals/CartPole-v0")
expert = PPO(policy=MlpPolicy, env=env)
expert.learn(1000)

rollouts = rollout.rollout(
    expert,
    make_vec_env(
        "seals/CartPole-v0",
        rng=rng,
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

venv = make_vec_env("seals/CartPole-v0", rng=rng, n_envs=8)
learner = PPO(env=venv, policy=MlpPolicy)
reward_net = BasicShapedRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm,
)
airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)
airl_trainer.train(20000)
rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
print("Rewards:", np.mean(rewards))
