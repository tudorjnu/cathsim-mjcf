
from cathsim import Navigate, Tip, Guidewire, Phantom

from dm_control import composer
from dm_control.suite.wrappers.pixels import Wrapper

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from gym import core, spaces
from gym.wrappers import TimeLimit

import cv2


try:
    from dm_env import specs
except ImportError:
    specs = None
try:
    # Suppress MuJoCo warning (dm_control uses absl logging).
    import absl.logging

    absl.logging.set_verbosity("error")
    from dm_control import suite
except (ImportError, OSError):
    suite = None
import numpy as np


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int16(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMEnv(core.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        env,
        env_kwargs=None,
        visualize_reward=False,
        from_pixels=False,
        height=256,
        width=256,
        camera_id=0,
        frame_skip=4,
        environment_kwargs=None,
        render_kwargs=None,
        channels_first=True,
        preprocess=True,
    ):

        self._render_kwargs = render_kwargs
        if render_kwargs is None:
            render_kwargs = {}

        self._height = render_kwargs.get('height', 256)
        self._width = render_kwargs.get('width', 256)
        self._camera_id = render_kwargs.get('camera_id', 0)

        self._from_pixels = from_pixels
        self._frame_skip = frame_skip
        self._channels_first = channels_first
        self.preprocess = preprocess
        self.render_mode = "rgb_array"

        if specs is None:
            raise RuntimeError(
                (
                    "The `specs` module from `dm_env` was not imported. Make sure "
                    "`dm_env` is installed and visible in the current python "
                    "environment."
                )
            )
        if suite is None:
            raise RuntimeError(
                (
                    "The `suite` module from `dm_control` was not imported. Make "
                    "sure `dm_control` is installed and visible in the current "
                    "python enviornment."
                )
            )

        # MDP creation
        self._env = env
        if from_pixels:
            self._env = Wrapper(self._env, render_kwargs=render_kwargs)
            print('Wrapped Env')

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [
                height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
            if preprocess:
                self._observation_space = spaces.Box(
                    low=-0.5, high=0.5, shape=shape, dtype=np.float32
                )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        self._state_space = _spec_to_box(self._env.observation_spec().values())

        self.current_state = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height, width=self._width, camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
            if self.preprocess:
                obs = obs / 255.0 - 0.5
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        done = False
        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, done, extra

    def reset(self, *, seed=None, options=None):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode="rgb_array", height=480, width=480, camera_id=0):
        assert mode == "rgb_array", "only support for rgb_array mode"
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        img = self._env.physics.render(
            height=height, width=width, camera_id=camera_id)
        return img


if __name__ == "__main__":
    env = DMEnv()
    env = TimeLimit(env, 100)

    print(env.observation_space.shape)
    obs = env.reset()
    print(obs.shape)

    # check_env(env)
    # exit()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    env.close()
