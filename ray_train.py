from datetime import timedelta

import numpy as np


from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.mbmpo.mbmpo import MBMPOConfig
from ray.rllib.algorithms.dreamer import DreamerConfig
from ray.tune.registry import register_env
from ray.rllib.utils import check_env

from wrapper import DMEnv, MPO
from gymnasium.wrappers import TimeLimit, FrameStack, RecordVideo

from dm_control import composer
from dm_control.suite.wrappers.pixels import Wrapper
from cathsim import Navigate, Tip, Guidewire, Phantom


def env_creator(env_config=None):
    phantom = Phantom("assets/phantom3.xml", model_dir="./assets")
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        dense_reward=False,
    )
    env = composer.Environment(
        task=task,
        time_limit=200,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )
    render_kwargs = {'width': 64, 'height': 64}
    env = DMEnv(
        env=env,
        from_pixels=False,
        time_limit=300,
        render_kwargs=render_kwargs,
    )
    # env = MPO(
    #     env=env,
    #     from_pixels=False,
    #     time_limit=300,
    #     render_kwargs=render_kwargs,
    #     use_image=False,
    #     channels_first=True,
    #     grayscale=True,
    #     preprocess=True
    # )
    env = TimeLimit(env, max_episode_steps=200)
    # env = FrameStack(env, 4)
    # env = RecordVideo(env, video_folder='./videos')
    return env


# env = env_creator()
# check_env(env)

register_env("cathsim", env_creator)

algo = (
    MBMPOConfig()
    .environment(env="cathsim")
    .resources(num_gpus=1)
    .framework("torch")
    # .training(
    #     model={
    #         # == LSTM ==
    #         # Whether to wrap the model with an LSTM.
    #         "use_lstm": True,
    #         # Max seq len for training the LSTM, defaults to 20.
    #         "max_seq_len": 20,
    #         # Size of the LSTM cell.
    #         "lstm_cell_size": 256,
    #         # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
    #         "lstm_use_prev_action": True,
    #         # Whether to feed r_{t-1} to LSTM.
    #         "lstm_use_prev_reward": True,
    #         # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
    #         "_time_major": False,
    #     }
    # )
    # .exploration(
    #     exploration_config={
    #         "type": "RE3",
    #         "embeds_dim": 128,
    #         "beta_schedule": "constant",
    #         "sub_exploration": {
    #             "type": "StochasticSampling",
    #         }
    #     }
    # )
    .build()
)
# ppo latest checkpoint
# algo.restore('/home/tudorjnu/ray_results/PPO_cathsim_2023-01-06_19-39-05_q_404mc/checkpoint_000091')
# sac latest checkpoint
# algo.restore('/home/tudorjnu/ray_results/SAC_cathsim_2023-01-07_16-21-42w6k4pngr/checkpoint_000091', fallback_to_latest=True)

for i in range(1000):
    result = algo.train()
    print(f'Iteration {result["training_iteration"]}: ({timedelta(seconds=round(result["time_this_iter_s"]))}s)')
    print('\tReward:', round(result['episode_reward_mean'], 2))
    print('\tLength', result['episode_len_mean'])

    if i % 20 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

print("finished")
