from pathlib import Path
from utils import make_env

from imitation.algorithms.bc import reconstruct_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import SAC, PPO


if __name__ == "__main__":
    expert_path = Path.cwd() / 'rl' / 'sb3' / 'trial_1'
    log_path = expert_path / 'logs' / 'bc'
    model_path = expert_path / 'checkpoints'
    for path in [log_path, model_path]:
        path.mkdir(parents=True, exist_ok=True)

    policy = reconstruct_policy(model_path / 'sac_bc')

    env = make_env(
        flatten_obs=True,
        time_limit=500,
        normalize_obs=False,
        frame_stack=1,
        render_kwargs=None,
        gym_version='gym',
        wrap_monitor=True,
        env_kwargs=dict(dense_reward=False),
    )

    model = SAC(policy='MlpPolicy', env=env, verbose=1)
    model.policy = policy
    model.learn(total_timesteps=10000, progress_bar=True)
