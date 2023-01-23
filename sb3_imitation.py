
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

env = gym.make("CartPole-v1")
rng = np.random.default_rng(0)


def train_expert():
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(100)  # Note: change this to 100000 to train a decent expert.
    return expert


def sample_expert_transitions():
    expert = train_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    print("Length Rollouts:", len(rollouts))
    print("Rollout Type:", rollouts[0])

    return rollout.flatten_trajectories(rollouts)


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


process_transitions(data)
transitions = sample_expert_transitions()
print("Transitions Type:", type(transitions))
print("Transitions Length:", len(transitions))
print("Transitions[0] Type:", type(transitions[0]))
print(transitions[0])
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
    render=True,
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
