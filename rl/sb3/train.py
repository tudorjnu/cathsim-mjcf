
from utils import train

EXP_NAME = "test"
ALGO = "ppo"

if __name__ == "__main__":
    train(
        algo=ALGO,
        experiment=EXP_NAME,
        time_steps=500_000,
        evaluate=True,
        env_kwargs={},
    )
