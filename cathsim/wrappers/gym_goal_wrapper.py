import numpy as np

import gym
from gym import spaces

from dm_env import specs


def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=dm_control_space.minimum,
                           high=dm_control_space.maximum,
                           dtype=np.float32)
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'),
                           high=float('inf'),
                           shape=dm_control_space.shape,
                           dtype=np.float32)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                             for key, value in dm_control_space.items()})
        return space


class DMEnv(gym.GoalEnv):
    def __init__(self, env, env_kwargs: dict = {}, render_kwargs: dict = {}):

        self._env = env
        self.metadata = {'render.modes': ['rgb_array'],
                         'video.frames_per_second': round(1.0 / self._env.control_timestep())}

        self.env_kwargs = env_kwargs
        self.render_kwargs = render_kwargs

        self.render_kwargs.get('camera_id', 0)
        self.render_kwargs.get('height', 256)
        self.render_kwargs.get('width', 256)

        self.action_space = convert_dm_control_to_gym_space(
            self._env.action_spec())
        self.observation_space = spaces.Dict(
            observation=convert_dm_control_to_gym_space(
                self._env.observation_spec()
            ),
            achieved_goal=spaces.Box(-1., 1., shape=(3,), dtype=np.float32),
            desired_goal=spaces.Box(-1., 1., shape=(3,), dtype=np.float32),
        )

        self.desired_goal = self._env.desired_goal

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

    def render(self, mode="rgb_array", height=256, width=256, camera_id=0):
        height = self.render_kwargs.get('height', height)
        width = self.render_kwargs.get('width', width)
        camera_id = self.render_kwargs.get('camera_id', camera_id)
        img = self._env.physics.render(height=height, width=width, camera_id=camera_id)
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self._env.close()

    def _get_obs(self, timestep):
        obs = timestep.observation
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs[key] = value.astype(np.float32)
        return obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self._env.compute_reward(achieved_goal, desired_goal)
        return reward


class Dict2Array(gym.ObservationWrapper):
    def __init__(self, env):
        super(Dict2Array, self).__init__(env)
        self.observation_space = next(iter(self.observation_space.values()))

    def observation(self, observation):
        obs = next(iter(observation.values()))
        return obs


if __name__ == "__main__":
    from cathsim.utils import make_env
    import cv2

    wrapper_kwargs = dict(
        use_pixels=True,
        use_obs=[
            'pixels',
        ],
        grayscale=True,
        resize_shape=128,
    )

    env = make_env(
        wrapper_kwargs=wrapper_kwargs
    )
    obs = env.reset()
    done = False
    print(obs.shape)
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        image = env.render()
        cv2.imshow('image', obs)
        cv2.waitKey(1)
