from pathlib import Path
import numpy as np

from torch.utils.data import Dataset


def process_trajectory(data):
    from imitation.data.types import Trajectory
    acts = data.pop('action')[:-1]
    obs = []
    for key, value in data.items():
        obs.append(value)
    obs = np.concatenate(obs, axis=1)
    return Trajectory(obs, acts, None, terminal=True)


def process_transitions(trial_path: str, images: bool = False):
    from imitation.data.rollout import flatten_trajectories
    from matplotlib import pyplot as plt
    trial_path = Path(trial_path)
    trajectories = []
    for episode_path in trial_path.iterdir():
        print('Processing: ', episode_path)
        data = np.load(episode_path / "trajectory.npz", allow_pickle=True)
        data = dict(data)
        if images:
            data.setdefault('pixels', [])
            images_path = episode_path / "images"
            for image_path in images_path.iterdir():
                data['pixels'].append(plt.imread(image_path))
        trajectories.append(process_trajectory(data))
    transitions = flatten_trajectories(trajectories)
    print(
        f'Processed {len(trajectories)} trajectories ({len(transitions)} transitions)')
    trajectory_lengths = [len(traj) for traj in trajectories]
    print('mean trajectory length:', np.mean(trajectory_lengths))

    return transitions


def process_image_transitions(trial_path: str):
    from imitation.data.rollout import flatten_trajectories
    from matplotlib import pyplot as plt
    trial_path = Path(trial_path)
    trajectories = []
    for episode_path in trial_path.iterdir():
        trajectory = {}
        data = np.load(episode_path / "trajectory.npz", allow_pickle=True)
        data = dict(data)
        trajectory['action'] = data['action']
        trajectory.setdefault('pixels', [])
        images_path = episode_path / "images"
        for image_path in images_path.iterdir():
            image = plt.imread(image_path)[:, :, 0]
            image = np.stack([image, image, image], axis=-1)
            image = image * 255
            image = image.astype(np.uint8)
            trajectory['pixels'].append(image)
        trajectories.append(process_trajectory(trajectory))
    transitions = flatten_trajectories(trajectories)
    print(
        f'Processed {len(trajectories)} trajectories ({len(transitions)} transitions)')
    trajectory_lengths = [len(traj) for traj in trajectories]
    print('mean trajectory length:', np.mean(trajectory_lengths))

    return transitions


class TransitionsDataset(Dataset):

    def __init__(self, trial_path: Path, transform=None):
        self.transitions = process_image_transitions(trial_path)
        self.transform = transform

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, index):
        transition = self.transitions[index]
        obs = transition['obs'].copy()
        obs = obs.astype(np.uint8)
        act = transition['acts'].copy()
        act = act.astype(np.float32)
        if self.transform:
            obs = self.transform(obs)
        return obs, act
