import numpy as np

from gym import spaces
import gym

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


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMEnv(gym.Env):
    def __init__(self, env, env_kwargs: dict = {}, render_kwargs: dict = {}):

        self.env = env
        self.metadata = {'render.modes': ['rgb_array'],
                         'video.frames_per_second': round(1.0 / self.env.control_timestep())}

        self.env_kwargs.get('from_pixels', False)
        self.env_kwargs.get('channels_first', True)
        self.env_kwargs.get('preprocess', False)

        self.render_kwargs.get('camera_id', 0)
        self.render_kwargs.get('height', 256)
        self.render_kwargs.get('width', 256)

        self.action_space = convert_dm_control_to_gym_space(
            self.env.action_spec())

        self.viewer = None

        if self.env_kwargs['from_pixels']:
            from dm_control.suite.wrappers import pixels
            self.env = pixels.Wrapper(
                self.env, render_kwargs=self.render_kwargs)
            shape = [3, self.render_kwargs['height'], self.render_kwargs['width']] if \
                env_kwargs['channels_first'] else \
                [self.render_kwargs['height'], self.render_kwargs['width'], 3]
            self.observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
            if env_kwargs['preprocess']:
                self.observation_space = spaces.Box(
                    low=-0.5, high=0.5, shape=shape, dtype=np.float32
                )
        else:
            self.observation_space = convert_dm_control_to_gym_space(
                self.env.observation_spec())

    def seed(self, seed):
        return self.env.random_state.seed

    def step(self, action):
        timestep = self.env.step(action)
        observation = self._get_obs(timestep)
        reward = timestep.reward
        done = timestep.last()
        info = {}
        return observation, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        obs = self._get_obs(timestep)
        return obs

    def render(self, mode="rgb_array", height=256, width=256, camera_id=0):
        assert mode == "rgb_array", "only support for rgb_array mode"
        height = self.render_kwargs.get('height', height)
        width = self.render_kwargs.get('width', width)
        camera_id = self.render_kwargs.get('camera_id', camera_id)
        img = self.env.physics.render(
            height=height, width=width, camera_id=camera_id)
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()

    def _get_obs(self, timestep):
        if self.env_kwargs['from_pixels']:
            obs = timestep.observation['pixels']
            # it doesn't seem that it works, can you make it work?
            if self.env_kwargs['channels_first']:
                obs = obs.transpose(2, 0, 1).copy()
            if self.env_kwargs['preprocess']:
                obs = obs / 255.0 - 0.5
                obs = obs.astype(np.float32)
        else:
            obs = timestep.observation
        return obs


if __name__ == "__main__":
    from cathsim.utils import make_env
    from gym.utils.env_checker import check_env

    env = make_env(
        flatten_obs=False,
        env_kwargs=dict(from_pixels=True, channels_first=True, preprocess=False),
    )

    print(env.observation_space.shape)
    obs = env.reset()
    print(obs.shape == env.observation_space.shape)

    check_env(env)