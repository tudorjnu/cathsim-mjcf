import numpy as np
import cv2

import gym
from gym import spaces

from dm_env import specs


def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        if len(dm_control_space.shape) > 1:
            space = spaces.Box(low=0,
                               high=255,
                               shape=dm_control_space.shape,
                               dtype=dm_control_space.dtype)
        else:
            space = spaces.Box(low=dm_control_space.minimum,
                               high=dm_control_space.maximum,
                               shape=dm_control_space.shape,
                               dtype=np.float32)
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'),
                           high=float('inf'),
                           shape=dm_control_space.shape,
                           dtype=np.float32)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict()
        for key, value in dm_control_space.items():
            space[key] = convert_dm_control_to_gym_space(value)
        return space


class DMEnv(gym.Env):
    def __init__(self, env, env_kwargs: dict = {}):

        self._env = env
        self.metadata = {'render.modes': ['rgb_array'],
                         'video.frames_per_second': round(1.0 / self._env.control_timestep())}

        self.env_kwargs = env_kwargs
        self.image_size = self._env.task.image_size

        self.action_space = convert_dm_control_to_gym_space(
            self._env.action_spec())
        self.observation_space = convert_dm_control_to_gym_space(
            self._env.observation_spec())

        self.viewer = None

    def seed(self, seed):
        return self._env.random_state.seed

    def step(self, action):
        timestep = self._env.step(action)
        observation = self._get_obs(timestep)
        reward = timestep.reward
        done = timestep.last()
        info = dict(
            head_pos=self._env._task.head_pos
        )
        return observation, reward, done, info

    def reset(self):
        timestep = self._env.reset()
        obs = self._get_obs(timestep)
        return obs

    def render(self, mode="rgb_array", image_size=None):
        image_size = image_size if image_size else self.image_size
        img = self._env.physics.render(
            height=image_size, width=image_size, camera_id=0)
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self._env.close()

    def _get_obs(self, timestep):
        obs = timestep.observation
        for key, value in obs.items():
            if value.dtype == np.float64:
                obs[key] = value.astype(np.float32)
        return obs


class Dict2Array(gym.ObservationWrapper):
    def __init__(self, env):
        super(Dict2Array, self).__init__(env)
        self.observation_space = next(iter(self.observation_space.values()))

    def observation(self, observation):
        obs = next(iter(observation.values()))
        return obs


class MultiInputImageWrapper(gym.ObservationWrapper):

    def __init__(
        self,
        env: gym.Env,
        grayscale: bool = False,
        keep_dim: bool = True,
        image_key: str = 'pixels',
    ):
        super(MultiInputImageWrapper, self).__init__(env)
        self.grayscale = grayscale
        self.keep_dim = keep_dim
        self.image_key = image_key

        image_space = self.observation_space.spaces[self.image_key]

        assert (
            len(image_space.shape) == 3
            and image_space.shape[-1] == 3
        ), "Image should be in RGB format"

        if self.grayscale:
            if self.keep_dim:
                image_space = spaces.Box(
                    low=0, high=255,
                    shape=(image_space.shape[0], image_space.shape[1], 1),
                    dtype=image_space.dtype
                )
            else:
                image_space = spaces.Box(
                    low=0, high=255,
                    shape=(image_space.shape[0], image_space.shape[1]),
                    dtype=image_space.dtype,
                )

        self.observation_space[self.image_key] = image_space

    def observation(self, observation):
        image = observation[self.image_key]
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.keep_dim:
                image = np.expand_dims(image, axis=-1)
        observation[self.image_key] = image
        return observation


if __name__ == "__main__":
    from gym.utils.env_checker import check_env
    from cathsim.env import Phantom, Tip, Guidewire, Navigate
    from dm_control import composer

    task_kwargs = dict(
        use_pixels=True,
        use_segment=True,
        image_size=30,
    )

    wrapper_kwargs = dict(
        grayscale=True,
    )

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
    )

    env = MultiInputImageWrapper(
        env,
        grayscale=wrapper_kwargs.get('grayscale', False),
        image_key=wrapper_kwargs.get('image_key', 'pixels'),
    )

    check_env(env)
