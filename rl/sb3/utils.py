import os
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC

from dm_control.viewer.application import Application
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL

ALGOS = {
    'airl': AIRL,
    'gail': GAIL,
    'ppo': PPO,
    'sac': SAC,
}


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


def process_image_transitions(trial_path: str, images: bool = False):
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
            trajectory['pixels'].append(image)
        trajectories.append(process_trajectory(trajectory))
    transitions = flatten_trajectories(trajectories)
    print(
        f'Processed {len(trajectories)} trajectories ({len(transitions)} transitions)')
    trajectory_lengths = [len(traj) for traj in trajectories]
    print('mean trajectory length:', np.mean(trajectory_lengths))

    return transitions


def make_experiment(experiment_name='trial_0'):
    from pathlib import Path
    experiment_path = Path(__file__).parent / 'experiments' / experiment_name
    model_path = experiment_path / 'models'
    eval_path = experiment_path / 'eval'
    log_path = experiment_path / 'logs'
    for dir in [experiment_path, model_path, log_path, eval_path]:
        dir.mkdir(parents=True, exist_ok=True)
    return model_path, log_path, eval_path


def make_env(env_kwargs: dict = {}, monitor_wrapper=False,) -> Callable:
    import gym
    from cathsim.utils import make_env

    def _init() -> gym.Env:
        env = make_env(**env_kwargs)
        if monitor_wrapper:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)
        return env
    return _init


def make_vec_env(num_env: int = None,
                 monitor_wrapper=True,
                 monitor_kwargs: dict = {},
                 env_kwargs: dict = {},
                 **kwargs):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    if num_env is None:
        num_env = os.cpu_count() // 2
    vec_env = SubprocVecEnv([make_env(**kwargs) for _ in range(num_env)])
    if monitor_wrapper:
        print('Monitor wrapper enabled')
        from stable_baselines3.common.vec_env import VecMonitor
        vec_env = VecMonitor(vec_env)
    return vec_env


def train(algo: str,
          indice: int,
          experiment: str,
          time_steps: int = 500_000,
          evaluate: bool = True,
          device: str = 'cpu',
          n_envs: int = None,
          vec_env: bool = True,
          env_kwargs: dict = {},
          **kwargs):
    n_envs = n_envs or os.cpu_count() // 2
    model_path, log_path, eval_path = make_experiment(experiment)

    if vec_env:
        env = make_vec_env(n_envs, env_kwargs=env_kwargs, monitor_wrapper=True)
    else:
        from cathsim.utils import make_env
        env = make_env(env_kwargs)

    if (model_path / f'{algo}.zip').exists():
        print(f'Loading {algo} model from {experiment} experiment.')
        model = ALGOS[algo].load(model_path / f'{algo}.zip')
    else:
        print(f'Training {algo} model in {experiment} experiment.')
        model = ALGOS[algo]('MlpPolicy',
                            env,
                            device=device,
                            verbose=1,
                            tensorboard_log=log_path,
                            **kwargs)

    model.learn(total_timesteps=time_steps,
                tb_log_name=f'{algo}_{indice}',
                progress_bar=True)
    model.save(model_path / f'{algo}_{indice}.zip')

    if evaluate:
        rewards, lengths, success_rate = eval_policy(model, env, 2, eval_path)
        np.savez(eval_path / f'{algo}_{indice}',
                 rewards=rewards, lengths=lengths)


def visualize_agent(algo, experiment):
    import cv2
    from cathsim.utils import make_env
    model_path, log_path, eval_path = make_experiment(experiment)
    model = ALGOS[algo].load(model_path / f'{algo}.zip')
    env = make_env()
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        image = env.render('rgb_array')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', image)
        cv2.waitKey(1)


class Application(Application):

    def __init__(self, title, width, height, trial_path=None):
        super().__init__(title, width, height)
        from dm_control.viewer import user_input

        self._input_map.bind(self._move_forward, user_input.KEY_UP)
        self._input_map.bind(self._move_back, user_input.KEY_DOWN)
        self._input_map.bind(self._move_left, user_input.KEY_LEFT)
        self._input_map.bind(self._move_right, user_input.KEY_RIGHT)
        self.null_action = np.zeros(2)
        self._step = 0
        self._episode = 0
        if trial_path.exists():
            episode_paths = sorted(trial_path.iterdir())
            if len(episode_paths) > 0:
                episode_num = episode_paths[-1].name.split('_')[-1]
                self._episode = int(episode_num) + 1
        self._policy = None
        self._trajectory = {}
        self._trial_path = trial_path
        self._episode_path = self._trial_path / f'episode_{self._episode:02}'
        self._images_path = self._episode_path / 'images'
        self._images_path.mkdir(parents=True, exist_ok=True)

    def _save_transition(self, observation, action):
        for key, value in observation.items():
            if key != 'top_camera':
                self._trajectory.setdefault(key, []).append(value)
            else:
                image_path = self._images_path / f'{self._step:03}.png'
                plt.imsave(image_path.as_posix(), value)
        self._trajectory.setdefault('action', []).append(action)

    def _initialize_episode(self):
        trajectory_path = self._episode_path / 'trajectory'
        np.savez_compressed(trajectory_path.as_posix(), **self._trajectory)
        self._restart_runtime()
        print(f'Episode {self._episode:02} finished')
        self._trajectory = {}
        self._step = 0
        self._episode += 1
        # change the episode path to the new episode
        self._episode_path = self._trial_path / f'episode_{self._episode:02}'
        self._images_path = self._episode_path / 'images'
        self._images_path.mkdir(parents=True, exist_ok=True)

    def perform_action(self):
        print(f'step {self._step:03}')
        time_step = self._runtime._time_step
        if not time_step.last():
            self._advance_simulation()
            action = self._runtime._last_action
            self._save_transition(time_step.observation, action)
            self._step += 1
        else:
            self._initialize_episode()

    def _move_forward(self):
        self._runtime._default_action = [1, 0]
        self.perform_action()

    def _move_back(self):
        self._runtime._default_action = [-1, 0]
        self.perform_action()

    def _move_left(self):
        self._runtime._default_action = [0, -1]
        self.perform_action()

    def _move_right(self):
        self._runtime._default_action = [0, 1]
        self.perform_action()


def launch(environment_loader, policy=None, title='Explorer', width=1024,
           height=768, trial_path=None):
    app = Application(title=title, width=width, height=height,
                      trial_path=trial_path)
    app.launch(environment_loader=environment_loader, policy=policy)


def record_expert_trajectories(trial_name: Path):
    from cathsim.env import Phantom, Tip, Guidewire, Navigate
    from dm_control import composer

    trial_path = Path(__file__).parent.parent / 'expert' / trial_name
    try:
        trial_path.mkdir(parents=True)
    except FileExistsError:
        cont_training = input(
            f'Trial {trial_name} already exists. Continue? [y/N] ')
        cont_training = 'n' if cont_training == '' else cont_training
        if cont_training.lower() == 'y':
            pass
        else:
            print('Aborting')
            exit()

    phantom = Phantom()
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_image=True,
    )
    env = composer.Environment(
        task=task,
        time_limit=200,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    action_spec_name = '\t' + env.action_spec().name.replace('\t', '\n\t')
    print('\nAction Spec:\n', action_spec_name)
    time_step = env.reset()
    print('\nObservation Spec:')
    for key, value in time_step.observation.items():
        print('\t', key, value.shape)

    launch(env, trial_path=trial_path)


def eval_policy(model, env, n_eval_episodes, eval_path, **kwargs):
    from stable_baselines3.common.evaluation import evaluate_policy

    rewards, lengths = evaluate_policy(model, env, n_eval_episodes,
                                       return_episode_rewards=True, **kwargs)
    print(f'Average reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}')
    print(f'Average length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}')
    success_rate = np.sum(lengths < np.max(lengths)) / len(lengths)
    print(f'Success rate: {success_rate:.2f}')
    return rewards, lengths, success_rate


def cmd_record_traj(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--trial_name', type=str, default='test')
    args = parser.parse_args(args)
    record_expert_trajectories(Path(args.trial_name))


if __name__ == '__main__':
    visualize_agent('sac', 'test')
