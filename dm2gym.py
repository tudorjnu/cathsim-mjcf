import gym
import cv2
from gym import spaces
from cathsim import Navigate, Tip, Guidewire, Phantom
from dm_control import composer
import numpy as np
from dm_env import specs


def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=dm_control_space.minimum,
                           high=dm_control_space.maximum,
                           dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'),
                           high=float('inf'),
                           shape=dm_control_space.shape,
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                             for key, value in dm_control_space.items()})
        return space


class DMEnv(gym.Env):
    def __init__(self, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
        phantom = Phantom("assets/phantom3.xml", model_dir="./assets")
        tip = Tip(n_bodies=4)
        guidewire = Guidewire(n_bodies=80)
        task = Navigate(
            phantom=phantom,
            guidewire=guidewire,
            tip=tip,
        )
        self.env = composer.Environment(
            task=task,
            time_limit=2000,
            random_state=np.random.RandomState(42),
            strip_singleton_obs_buffer_dim=True,
        )
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0 / self.env.control_timestep())}

        self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.max_episode_steps = self.env._time_limit
        self.viewer = None

    def seed(self, seed):
        return self.env.task.random.seed(seed)

    def step(self, action):
        timestep = self.env.step(action)
        observation = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        info = {}
        return observation, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        return timestep.observation

    def render(self, mode='rgb_array', **kwargs):
        if 'camera_id' not in kwargs:
            kwargs['camera_id'] = 0  # Tracking camera
        use_opencv_renderer = kwargs.pop('use_opencv_renderer', False)

        # img = self.env.physics.render(**kwargs)
        img = self.env.physics.render(camera_id=0, height=200, width=200)
        if mode == 'rgb_array':
            cv2.imshow("Top Camera", img)
            cv2.waitKey(1)
            return img
        elif mode == 'human':
            if self.viewer is None:
                if not use_opencv_renderer:
                    from gym.envs.classic_control import rendering
                    self.viewer = rendering.SimpleImageViewer(maxwidth=1024)
                else:
                    from . import OpenCVImageViewer
                    self.viewer = OpenCVImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()
