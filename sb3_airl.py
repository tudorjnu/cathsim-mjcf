import gym
import numpy as np
from pathlib import Path
from utils import make_env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy

from imitation.data.wrappers import RolloutInfoWrapper

from utils import process_transitions
from sb3_algos import ALGOS


from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

if __name__ == "__main__":
    trial_path = Path.cwd() / 'rl' / 'sb3' / 'trial_1'
    log_path = trial_path / 'logs'
    model_path = trial_path / 'checkpoints'

    for path in [log_path, model_path]:
        path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

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

    trial_path = Path.cwd() / "rl" / "expert" / "trial_1"
    transitions = process_transitions(trial_path)

    venv = make_vec_env(lambda: make_env(), n_envs=8)
    learner = PPO(env=venv, policy=MlpPolicy)
    reward_net = BasicShapedRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    trainer = ALGOS['airl'](
        demonstrations=transitions,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    trainer.train(100000)
    rewards, lengths = evaluate_policy(
        learner, venv, 4, return_episode_rewards=True)
    print(f"Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Mean length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")
    trainer.policy.save(model_path / 'airl')
