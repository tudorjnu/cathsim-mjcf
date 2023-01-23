from typing import Callable
from wrapper_gym import DMEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from gym.wrappers import TimeLimit, RecordVideo
from dm_control import composer
import gym

import numpy as np

from cathsim import Navigate, Tip, Guidewire, Phantom


def env_creator():
    render_kwargs = {'width': 64, 'height': 64}
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
        from_pixels=False,
        render_kwargs=render_kwargs,
        channels_first=True,
    )
    env = TimeLimit(env, max_episode_steps=400)
    # env = FrameStack(env, 4)
    return env


def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = env_creator()
        env.seed(seed + rank)
        return env
    return _init


if __name__ == "__main__":
    env = SubprocVecEnv([make_env(i) for i in range(4)])
    obs = env.reset()
    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=1000)
