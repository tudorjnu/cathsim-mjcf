import gym
import numpy as np
from pathlib import Path
from utils import make_env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper

from utils import process_transitions
from sb3_algos import ALGOS


from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    env = make_env(
        flatten_obs=True,
        time_limit=200,
        normalize_obs=False,
        frame_stack=1,
        render_kwargs=None,
        env_kwargs=None,
        gym_version='gym',
        wrap_monitor=True,
    )

    trial_path = Path.cwd() / "rl" / "expert" / "trial_0"
    transitions = process_transitions(trial_path)

    venv = make_vec_env(lambda: make_env(), n_envs=8)
    learner = PPO(env=venv, policy=MlpPolicy)

    reward_net = BasicShapedRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    airl_trainer = ALGOS['airl'](
        demonstrations=transitions,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    airl_trainer.train(20000)
    rewards, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True)

    print("Rewards:", rewards)
