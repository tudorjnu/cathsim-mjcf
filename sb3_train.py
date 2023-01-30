from pathlib import Path
from utils import make_env

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    trial_path = Path.cwd() / 'rl' / 'sb3' / 'trial_0'
    log_path = trial_path / 'logs'
    model_path = trial_path / 'checkpoints'
    for path in [log_path, model_path]:
        path.mkdir(parents=True, exist_ok=True)

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

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=trial_path.as_posix(),
        name_prefix="PPO",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )

    env = make_vec_env(lambda: make_env(), n_envs=4,
                       vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy", env, verbose=1, n_steps=512, device='cuda',
                tensorboard_log=trial_path.as_posix(), seed=42)
    checkpoint_path = trial_path / 'PPO_96000_steps.zip'
    PPO.load(checkpoint_path.as_posix(), env=env)
    model.learn(total_timesteps=1e5, progress_bar=True,
                callback=checkpoint_callback, reset_num_timesteps=True)
