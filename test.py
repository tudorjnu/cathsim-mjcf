from ray.rllib.algorithms.mbmpo import MBMPOConfig

config = MBMPOConfig()

config = config.training(lr=0.0003, train_batch_size=512)

config = config.resources(num_gpus=1)

config = config.rollouts(num_rollout_workers=64)
config.framework("torch")

print(config.to_dict())

# Build a Algorithm object from the config and run 1 training iteration.

algo = config.build(env="CartPole-v1")

algo.train()
