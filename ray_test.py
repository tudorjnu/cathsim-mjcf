from dm_control import composer
from moviepy.editor import *
from gymnasium.wrappers import TimeLimit, FrameStack
import cv2
import numpy as np

from cathsim import Navigate, Tip, Guidewire, Phantom
from wrapper import DMEnv

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from ray.rllib.utils import check_env


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
        time_limit=100,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )
    render_kwargs = {'width': 128, 'height': 128}
    env = DMEnv(
        env=env,
        from_pixels=False,
        time_limit=100,
        render_kwargs=render_kwargs,
        use_image=False,
        channels_first=False,
    )
    env = TimeLimit(env, max_episode_steps=200)
    env = FrameStack(env, 4)
    # env = RecordVideo(env, video_folder='./videos')
    return env


register_env("cathsim", env_creator)

algo = (
    PPOConfig()
    .environment(env="cathsim")
    .resources(num_gpus=1)
    # .framework("tf2")
    .exploration(
        exploration_config={
            "type": "RE3",
            "embeds_dim": 128,
            "beta_schedule": "constant",
            "sub_exploration": {
                "type": "StochasticSampling",
            }
        }
    )
    .build()
)

env = env_creator()

algo.restore('/home/tudorjnu/ray_results/PPO_cathsim_2023-01-09_21-41-27h8dcxi0k/checkpoint_000181')

obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0
i = 0
frames = []
while not terminated and not truncated:
    # for _ in range(200):
    img = env.render()
    frames.append(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Obs", img)
    cv2.waitKey(1)
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    i += 1

clips = [ImageClip(m).set_duration(0.2)
         for m in frames]

concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("test.mp4", fps=24)

print(f"total-reward={total_reward}")
print(f"episode_length={i}")
