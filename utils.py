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


def make_env(flatten_obs: bool = True, time_limit: int = 200,
             normalize_obs: bool = True, frame_stack: int = 3,
             render_kwargs: dict = None, env_kwargs: dict = None,
             gym_version: str = 'gym', wrap_monitor: bool = False):
    """
    Create a gym environment from cathsim, dm_control environment.

    :param flatten_obs: flattens the observation space
    :param time_limit: sets a time limit to the environment
    :param normalize_obs: normalizes the observation space
    :param frame_stack: stacks n frames of the environment
    :param render_kwargs: dict of kwargs for the render function. Valid keys are:
        from_pixels: bool, if True, render from pixels
        width: int, width of the rendered image
        height: int, height of the rendered image
        camera_id: int, camera id to use
    :param env_kwargs: dict of kwargs for the environment. Valid keys are:
        None for now
    :param gym_version: gyn or gymnasium 
    """
    from cathsim import Phantom, Tip, Guidewire, Navigate
    from dm_control import composer
    if gym_version == 'gym':
        from gym import wrappers
        from wrapper_gym import DMEnv
    elif gym_version == 'gymnasium':
        from gymnasium import wrappers
        from wrapper_gymnasium import DMEnv
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
        render_kwargs=render_kwargs,
        env_kwargs=env_kwargs,
    )
    if gym_version == 'gymnasium':
        env = wrappers.EnvCompatibility(env, render_mode='rgb_array')
    if flatten_obs:
        env = wrappers.FlattenObservation(env)
    if time_limit is not None:
        env = wrappers.TimeLimit(env, max_episode_steps=time_limit)
    if normalize_obs:
        env = wrappers.NormalizeObservation(env)
    if frame_stack > 1:
        env = wrappers.FrameStack(env, frame_stack)
    if wrap_monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
    return env
