from pathlib import Path
import numpy as np
from utils import make_env

from imitation.algorithms.bc import reconstruct_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC, PPO


if __name__ == "__main__":
    expert_path = Path.cwd() / 'rl' / 'sb3' / 'trial_1'
    log_path = expert_path / 'logs' / 'bc'
    model_path = expert_path / 'checkpoints'
    for path in [log_path, model_path]:
        path.mkdir(parents=True, exist_ok=True)

    policy = reconstruct_policy(model_path / 'bc_policy')

    env = make_env(
        flatten_obs=True,
        time_limit=500,
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

    rewards, lengths = evaluate_policy(
        policy,
        env,
        n_eval_episodes=30,
        return_episode_rewards=True,
    )

    print(f"Reward after training: {np.mean(rewards)}")
    print(f"Lengths: {np.mean(lengths)}")
    # count how many episodes have a length lower than 500
    success_rate = np.sum(np.array(lengths) < 500) / len(lengths)
    print('Success Rate: ', round(success_rate, 4))
