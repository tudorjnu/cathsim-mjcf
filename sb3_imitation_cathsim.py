"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import types

from wrapper_gym import DMEnv
from gym.wrappers import TimeLimit, RecordVideo
from dm_control import composer
from cathsim import Navigate, Tip, Guidewire, Phantom


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
    env = TimeLimit(env, max_episode_steps=400)
    # env = FrameStack(env, 4)
    return env


env = env_creator()

# load np pickle file
data = np.load("./data/episode_0/trajectory.npz", allow_pickle=True)


def process_transitions(data):
    obs = []
    acts = []
    print("Processing expert transitions.")
    # merge the values of the dictionary of the keys that are not "action"
    for key in data.keys():
        if key != "action":
            obs.append(data[key])
        else:
            acts.append(data[key])
    obs = np.concatenate(obs, axis=-1)
    new_obs = obs[1:]
    obs = obs[:-1]
    acts = np.concatenate(acts)[0:-1]
    dones = np.zeros_like(acts[:, 0])
    dones[-1] = 1
    dones = dones.astype(bool)
    print("Length of obs:", len(obs))
    print("Length of acts:", len(acts[:, 0]))
    return types.Transitions(obs, acts, [{} for i in obs], new_obs, dones=dones)


transitions = process_transitions(data)

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

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=True,
)
print(f"Reward after training: {reward}")
