
from wrapper_gym import DMCEnv
from stable_baselines3.common.env_checker import check_env
from gym.wrappers import TimeLimit, RecordVideo


def env_creator():
    env = DMCEnv()
    # env = TimeLimit(env, max_episode_steps=1000)
    # env = RecordVideo(env, video_folder='./videos')
    # env = FrameStack(env, 4)
    return env


env = env_creator()
print(env.observation_space.shape)
obs = env.reset()
print(obs.shape)
check_env(env)
