import numpy as np
import cv2
from pathlib import Path

from utils import make_env

from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.bc import reconstruct_policy

if __name__ == "__main__":
    expert_path = Path.cwd() / 'rl' / 'sb3' / 'trial_1'
    log_path = expert_path / 'logs' / 'bc'
    model_path = expert_path / 'checkpoints'
    for path in [log_path, model_path]:
        path.mkdir(parents=True, exist_ok=True)

    policy = reconstruct_policy(model_path / 'bc')

    env = make_env(
        flatten_obs=True,
        time_limit=200,
        normalize_obs=False,
        frame_stack=1,
        render_kwargs=None,
        gym_version='gym',
        wrap_monitor=True,
        env_kwargs=None,
        task_kwargs=dict(
            dense_reward=False,
        ),
    )

    rewards = []
    lengths = []
    for episode in range(10):
        print(f"Episode {episode}")
        episode_rewards = []
        obs = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            image = env.render(mode='rgb_array')
            cv2.imshow('image', image)
            cv2.waitKey(1)
        rewards.append(sum(episode_rewards))
        lengths.append(len(episode_rewards))

    print(f"Average reward: {np.mean(rewards)}")
    print(f"Average length: {np.mean(lengths)}")
