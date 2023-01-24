import numpy as np
from imitation.data import rollout, types
from pathlib import Path


def process_trajectory(data):
    acts = data.pop('action')[:-1]
    obs = []
    for key, value in data.items():
        obs.append(value)
    obs = np.concatenate(obs, axis=1)
    return types.Trajectory(obs, acts, None, terminal=True)


def process_transitions(trial_path: Path) -> types.Transitions:
    # traverse the directory and load the npz files
    # process each file into a trajectory
    trajectories = []
    print("Processing expert transitions.")
    for episode_path in trial_path.iterdir():
        data = np.load(episode_path / "trajectory.npz", allow_pickle=True)
        data = dict(data)
        trajectories.append(process_trajectory(data))
    print('Processed {} trajectories'.format(len(trajectories)))
    return rollout.flatten_trajectories(trajectories)
