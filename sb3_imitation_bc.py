import numpy as np
from pathlib import Path

from utils import process_transitions, make_env

from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc


if __name__ == "__main__":

    rng = np.random.default_rng(0)

    env = make_env(
        flatten_obs=True,
        time_limit=200,
        normalize_obs=False,
        frame_stack=1,
        render_kwargs=None,
        env_kwargs=None,
        gym_version='gym',
        wrap_monitor=True,
    )

    trial_path = Path.cwd() / "rl" / "expert" / "trial_1"
    transitions = process_transitions(trial_path)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=3,
    )
    print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=200)

    bc_trainer.save("./rl/checkpoint/bc_model")

    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=3,
    )

    print(f"Reward after training: {reward}")
