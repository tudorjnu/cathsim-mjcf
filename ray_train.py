from datetime import timedelta
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.utils import check_env

from wrapper import DMEnv
from gymnasium.wrappers import TimeLimit, FrameStack, RecordVideo, GrayScaleObservation

from dm_control import composer
from cathsim import Navigate, Tip, Guidewire, Phantom

from algorithms import CONFIGS


def env_creator(env_config=None):
    render_kwargs = {'width': 64, 'height': 64}
    phantom = Phantom("assets/phantom3.xml", model_dir="./assets")
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
        from_pixels=True,
        render_kwargs=render_kwargs,
        channels_first=True,
    )
    env = TimeLimit(env, max_episode_steps=400)
    # env = FrameStack(env, 4)
    # env = RecordVideo(env, video_folder='./videos')
    return env


# env = env_creator()
# check_env(env)

register_env("cathsim", env_creator)

algo = (
    CONFIGS["Dreamer"]()
    .environment(env="cathsim")
    .resources(num_gpus=1)
    .framework("torch")
    .build()
)

for i in range(1000):
    result = algo.train()
    print(
        f'Iteration {result["training_iteration"]}: ({timedelta(seconds=round(result["time_this_iter_s"]))}s)')
    print('\tReward:', round(result['episode_reward_mean'], 2))
    print('\tLength', result['episode_len_mean'])

    # if i % 20 == 0:
    # checkpoint_dir = algo.save()
    # print(f"Checkpoint saved in directory {checkpoint_dir}")

print("finished")
