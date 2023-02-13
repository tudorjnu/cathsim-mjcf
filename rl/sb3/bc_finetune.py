from utils import make_vec_env, make_experiment, ALGOS
from imitation.algorithms.bc import reconstruct_policy


ALGO = 'sac'

if __name__ == "__main__":
    model_path, log_path, eval_path = make_experiment("test")

    policy = reconstruct_policy(model_path / 'bc_baseline')

    vec_env = make_vec_env()

    model = ALGOS[ALGO](policy='MlpPolicy', env=vec_env, verbose=1, device='cpu',
                        tensorboard_log=log_path, seed=0)
    model.policy = policy
    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save(model_path / 'bc_finetuned.zip')
