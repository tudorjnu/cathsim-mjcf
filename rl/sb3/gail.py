from pathlib import Path

from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import BasicRewardNet

from stable_baselines3 import PPO

from utils import process_transitions
from utils import make_experiment, make_vec_env, eval_policy, ALGOS

if __name__ == "__main__":
    model_path, log_path, eval_path = make_experiment('test')
    transitions_path = Path.cwd() / 'rl' / 'expert' / 'trial_2'

    venv = make_vec_env()
    learner = PPO(
        env=venv,
        policy='MlpPolicy',
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        verbose=1,
        device='cpu',
    )
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    transitions = process_transitions(transitions_path)

    trainer = ALGOS['gail'](
        demonstrations=transitions,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    trainer.train(20000)  # Note: set to 300000 for better results
    learner.save(model_path / "gail_ppo")

    # Evaluate the trained agent
    eval_policy(learner, venv, 3, eval_path)
