import numpy as np
import matplotlib.pyplot as plt
from dm_control.viewer.application import Application


def make_env(flatten_obs: bool = True, time_limit: int = 300,
             normalize_obs: bool = True, frame_stack: int = 1,
             render_kwargs: dict = None, env_kwargs: dict = None,
             gym_version: str = 'gym', task_kwargs: dict = {}):
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
    from cathsim.env import Phantom, Tip, Guidewire, Navigate
    from dm_control import composer
    if gym_version == 'gym':
        from gym import wrappers
        from cathsim.wrappers.wrapper_gym import DMEnv
    elif gym_version == 'gymnasium':
        from gymnasium import wrappers
        from wrapper_gymnasium import DMEnv
    phantom = Phantom()
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        **task_kwargs,
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
    return env


class Application(Application):

    def __init__(self, title, width, height):
        super().__init__(title, width, height)
        from dm_control.viewer import user_input

        self._input_map.bind(self._move_forward, user_input.KEY_UP)
        self._input_map.bind(self._move_back, user_input.KEY_DOWN)
        self._input_map.bind(self._move_left, user_input.KEY_LEFT)
        self._input_map.bind(self._move_right, user_input.KEY_RIGHT)
        self.null_action = np.zeros(2)
        self._step = 0
        self._policy = None

    def _initialize_episode(self):
        self._restart_runtime()
        self._step = 0

    def perform_action(self):
        print(f'step {self._step:03}')
        time_step = self._runtime._time_step
        if not time_step.last():
            self._advance_simulation()
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
    app = Application(title=title, width=width, height=height)
    app.launch(environment_loader=environment_loader, policy=policy)


def run_env(args=None):
    from argparse import ArgumentParser
    from dm_control import composer
    from cathsim.env import Phantom, Tip, Guidewire, Navigate

    parser = ArgumentParser()
    parser.add_argument('--n_bodies', type=int, default=80)
    parser.add_argument('--tip_n_bodies', type=int, default=4)

    parsed_args = parser.parse_args(args)

    phantom = Phantom()
    tip = Tip(n_bodies=parsed_args.tip_n_bodies)
    guidewire = Guidewire(n_bodies=parsed_args.n_bodies)

    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    launch(env)


if __name__ == "__main__":
    run_env(args=None)
