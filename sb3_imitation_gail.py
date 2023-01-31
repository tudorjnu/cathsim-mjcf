import gym
import numpy as np
from pathlib import Path
from utils import make_env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper

from utils import process_transitions
from sb3_algos import ALGOS


from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

if __name__ == "__main__":
    expert_path = Path.cwd() / 'rl' / 'sb3' / 'trial_1'
    log_path = expert_path / 'logs'
    model_path = expert_path / 'checkpoints'

    for path in [log_path, model_path]:
        path.mkdir(parents=True, exist_ok=True)
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

    expert_path = Path.cwd() / "rl" / "expert" / "trial_1"
    transitions = process_transitions(expert_path)

    venv = make_vec_env(lambda: make_env(), n_envs=8)
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=50000,
    )

    reward_net = BasicShapedRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    airl_trainer = ALGOS['gail'](
        demonstrations=transitions,
        venv=venv,
        demo_batch_size=128,
        gen_replay_buffer_capacity=256,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=400,
        save_path=expert_path.as_posix(),
        name_prefix="GAIL",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )

    rewards, lengths = evaluate_policy(
        learner, venv, 10, return_episode_rewards=True)
    print("Before training:")
    print(f"\tMean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"\tMean length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")
    airl_trainer.train(50000)
    rewards, lengths = evaluate_policy(
        learner, venv, 10, return_episode_rewards=True)

    print("After training")
    print(f"\tMean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"\tMean length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")
    airl_trainer.save(model_path.as_posix())
