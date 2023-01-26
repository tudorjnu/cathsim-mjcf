"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""

import gym
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import types
from wrapper_gym import _flatten_obs

from wrapper_gym import DMEnv
from utils import process_transitions
from gym.wrappers import TimeLimit, RecordVideo
from dm_control import composer
from cathsim import Navigate, Tip, Guidewire, Phantom
from sb3_algos import ALGOS

env = gym.make("CartPole-v1")
rng = np.random.default_rng(0)


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
    env = TimeLimit(env, max_episode_steps=200)
    return env


env = env_creator()

trial_path = Path.cwd() / "rl" / "expert" / "trial_0"
transitions = process_transitions(trial_path)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

bc_trainer.save("./rl/checkpoint/bc_model")

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
)

print(f"Reward after training: {reward}")
