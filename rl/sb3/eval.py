import numpy as np
from sb3_utils import make_experiment, make_vec_env, ALGOS
from stable_baselines3.common.evaluation import evaluate_policy

EXP_NAME = "0"
MODEL_NAME = "ppo"
DEVICE = "cpu"

if __name__ == "__main__":
    model_dir, log_dir, eval_path = make_experiment(EXP_NAME)
    model = ALGOS[MODEL_NAME].load(model_dir / MODEL_NAME)

    vec_env = make_vec_env()
    rewards, lengths = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=30,
        return_episode_rewards=True,
    )

    print(f"Reward after training: {np.mean(rewards)}")
    print(f"Lengths: {np.mean(lengths)}")
    success_rate = np.sum(np.array(lengths) < 500) / len(lengths)
    print('Success Rate (%): ', round(success_rate * 100, 2))
    np.savez(eval_path / 'rewards', rewards=rewards, lengths=lengths)
