from dm_control import composer
import numpy as np
from cathsim import Navigate, Tip, Guidewire, Phantom
from ray.rllib.env.wrappers.dm_env_wrapper import DMEnv
from ray.rllib.algorithms.ppo import PPOConfig

phantom = Phantom("assets/phantom3.xml", model_dir="./assets")
tip = Tip(n_bodies=4)
guidewire = Guidewire(n_bodies=80)

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

print(env.action_spec())
print(env.observation_spec())

exit()
wrapped_env = DMEnv(env)
# Create an RLlib Algorithm instance from a PPOConfig to learn how to
# act in the above environment.
config = (
    PPOConfig()
    .environment(
        # Env class to use (here: our gym.Env sub-class from above).
        env=DMEnv(env),
        # Config dict to be passed to our custom env's constructor.
    )
    # Parallelize environment rollouts.
    .rollouts(num_rollout_workers=3)
)
# Use the config's `build()` method to construct a PPO object.
algo = config.build()

# Train for n iterations and report results (mean episode rewards).
# Since we have to guess 10 times and the optimal reward is 0.0
# (exact match between observation and action value),
# we can expect to reach an optimal episode reward of 0.0.
for i in range(5):
    results = algo.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
