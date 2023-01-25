from typing import Callable
from pathlib import Path

from cathsim import Navigate, Tip, Guidewire, Phantom
from wrapper_gym import DMEnv
from dm_control import composer

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
# from sb3_contrib.ppo_recurrent.policies import
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import gym
import numpy as np


def env_creator(flatten_obs: bool = True, time_limit: int = 200,
                normalize_obs: bool = True, frame_stack: int = 3,
                render_kwargs: dict = None, env_kwargs: dict = None):
    phantom = Phantom("assets/phantom4.xml", model_dir="./assets")
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
    )
    env = composer.Environment(
        task=task,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )
    env = DMEnv(
        env=env,
        render_kwargs=render_kwargs,
        env_kwargs=env_kwargs,
    )
    if flatten_obs:
        from gym.wrappers import FlattenObservation
        env = FlattenObservation(env)
    if time_limit is not None:
        from gym.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=time_limit)
    if normalize_obs:
        from gym.wrappers import NormalizeObservation
        env = NormalizeObservation(env)
    if frame_stack > 1:
        from gym.wrappers import FrameStack
        env = FrameStack(env, frame_stack)
    return env


if __name__ == "__main__":
    log_path = Path.cwd() / 'rl' / 'sb3' / 'logs' / 'trial_0'
    log_path.mkdir(parents=True, exist_ok=True)

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
    env = SubprocVecEnv([lambda: env_creator() for i in range(n_envs)])
    # env = env_creator()
    model = PPO("MlpPolicy", env, verbose=1, n_steps=512,
                tensorboard_log=log_path.as_posix(), seed=42)
    model.learn(total_timesteps=1e5, progress_bar=True,
                callback=checkpoint_callback, )
