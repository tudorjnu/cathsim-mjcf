import numpy as np
from pathlib import Path

from utils import process_transitions, make_env

from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc


if __name__ == "__main__":
    expert_path = Path.cwd() / 'rl' / 'sb3' / 'trial_1'
    log_path = expert_path / 'logs' / 'bc'
    model_path = expert_path / 'checkpoints'
    for path in [log_path, model_path]:
        path.mkdir(parents=True, exist_ok=True)
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
        n_eval_episodes=4,
    )
    print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=400)

    rewards, lengths = evaluate_policy(
        bc_trainer.policy,
        env,
        n_eval_episodes=4,
        return_episode_rewards=True,
    )

    print(f"Reward after training: {np.mean(rewards)}")
    print(f"Lengths: {np.mean(lengths)}")

    bc_trainer.save_policy(str(model_path / 'bc'))
