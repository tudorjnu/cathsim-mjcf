from typing import Callable
from pathlib import Path
from utils import make_env

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
# from sb3_contrib.ppo_recurrent.policies import
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import gym
import numpy as np


if __name__ == "__main__":
    log_path = Path.cwd() / 'rl' / 'sb3' / 'logs' / 'trial_0'
    log_path.mkdir(parents=True, exist_ok=True)

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

# Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=log_path.as_posix(),
        name_prefix="PPO",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )

    n_envs = 4
    env = make_vec_env(lambda: make_env(), n_envs=n_envs,
                       vec_env_cls=SubprocVecEnv)

    # env = env_creator()
    model = PPO("MlpPolicy", env, verbose=1, n_steps=512, device='cuda',
                tensorboard_log=log_path.as_posix(), seed=42)
    model.learn(total_timesteps=1e5, progress_bar=True,
                callback=checkpoint_callback, )
