from dm2gym import DMEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from ray.rllib.utils import check_env


def env_creator(env_config=None):
    return DMEnv()


env = env_creator()
check_env(env)


register_env("cathsim", env_creator)

algo = (
    PPOConfig()
    .framework("torch")
    .rollouts(num_rollout_workers=6, horizon=2000)
    .resources(num_gpus=1)
    .environment(env="cathsim")
    .build()
)

for i in range(200):
    result = algo.train()
    print('Episode Mean:', result['episode_reward_mean'])

    if i % 10 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

print("finished")
