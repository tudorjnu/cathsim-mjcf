from pathlib import Path
from utils import make_env
import os

from imitation.algorithms.bc import reconstruct_policy
from stable_baselines3.common.env_util import make_vec_env
from sb3_algos import ALGOS

N_CPU = os.cpu_count() // 2
print(f"Number of CPUs: {N_CPU}")

if __name__ == "__main__":
    expert_path = Path.cwd() / 'rl' / 'sb3' / 'trial_1'
    log_path = expert_path / 'logs' / 'bc'
    model_path = expert_path / 'checkpoints'
    for path in [log_path, model_path]:
        path.mkdir(parents=True, exist_ok=True)

    policy = reconstruct_policy(model_path / 'bc_policy')

    def make_env_fn():
        return make_env(
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
    vec_env = make_vec_env(make_env_fn, n_envs=N_CPU)

    model = ALGOS['ppo'](policy='MlpPolicy', env=vec_env, verbose=1)
    model.policy = policy
    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save(model_path / 'bc_finetuned')
