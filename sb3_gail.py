from pathlib import Path
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy

from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import BasicRewardNet

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

from utils import make_env
from utils import process_transitions
from sb3_algos import ALGOS

if __name__ == "__main__":
    trial_path = Path.cwd() / "rl" / "expert" / "trial_1"
    expert_path = Path.cwd() / 'rl' / 'sb3' / 'trial_1'
    log_path = expert_path / 'logs'
    model_path = expert_path / 'checkpoints'

    for path in [log_path, model_path]:
        path.mkdir(parents=True, exist_ok=True)

    env = make_env(
        flatten_obs=True,
        time_limit=300,
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

    venv = make_vec_env(lambda: make_env(), n_envs=8)
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        verbose=1,
    )
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    transitions = process_transitions(trial_path)

    trainer = ALGOS['gail'](
        demonstrations=transitions,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )
    trainer.train(20000)  # Note: set to 300000 for better results
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )

    print(np.mean(learner_rewards_after_training))
    print(np.mean(learner_rewards_before_training))

    learner.policy.save(model_path / "gail_ppo")
