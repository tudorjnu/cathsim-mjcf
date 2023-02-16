import numpy as np
import gym
from gym import spaces
from dm_env import specs
from dm_control import composer


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


class DMEnv(gym.GoalEnv):
    def __init__(self,
                 env: composer.Environment,
                 env_kwargs: dict = {},
                 ):

        self._env = env
        self.set_goal(np.array([-0.043272, 0.136586, 0.034102], dtype=np.float32))
        self.metadata = {'render.modes': ['rgb_array'],
                         'video.frames_per_second': round(1.0 / self._env.control_timestep())}

        self.env_kwargs = env_kwargs
        self.image_size = self._env.task.image_size

        self.action_space = convert_dm_control_to_gym_space(
            self._env.action_spec())
        self.observation_space = convert_dm_control_to_gym_space(
            self._env.observation_spec())

        self.observation_space = spaces.Dict({
            'observation': self.observation_space,
            'achieved_goal': spaces.Box(
                low=0,
                high=1,
                shape=(3,),
                dtype=np.float32
            ),
            'desired_goal': spaces.Box(
                low=0,
                high=1,
                shape=(3,),
                dtype=np.float32
            )
        })

        self.observation_space['achieved_goal'] = spaces.Box(
            low=0,
            high=1,
            shape=(3,),
            dtype=np.float32
        )

        self.observation_space['desired_goal'] = spaces.Box(
            low=0,
            high=1,
            shape=(3,),
            dtype=np.float32
        )

        self.viewer = None

    def seed(self, seed):
        return self._env.random_state.seed

    def step(self, action):
        timestep = self._env.step(action)
        observation = self._get_obs(timestep)
        reward = timestep.reward
        done = timestep.last()
        info = dict(
        )
        return observation, reward, done, info

    def reset(self):
        timestep = self._env.reset()
        obs = self._get_obs(timestep)
        return obs

    def render(self, mode="rgb_array"):
        img = self._env.physics.render(
            height=self.image_size, width=self.image_size, camera_id=0)
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
        obs = dict(
            observation=obs,
            achieved_goal=self.achieved_goal,
            desired_goal=self.desired_goal,
        )
        return obs

    @property
    def achieved_goal(self):
        return self._env.task._get_head_pos(self._env.physics).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self._env.task.compute_reward(achieved_goal, desired_goal)
        return reward

    @property
    def desired_goal(self):
        return self._env.task._target_pos

    def set_goal(self, goal):
        self._env.task._target_pos = goal


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

    obs = env.reset()
    for key, value in obs.items():
        if key != 'observation':
            print(key, value.shape, value.dtype)
        else:
            for k, v in value.items():
                print(k, v.shape, v.dtype)
    print('\n')
    for i in range(1):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print('achieved_goal', obs['achieved_goal'], obs['achieved_goal'].dtype, obs['achieved_goal'].shape)
        print('desired_goal', obs['desired_goal'], obs['desired_goal'].dtype, obs['desired_goal'].shape)

    check_env(env)
