from utils import train

EXP_NAME = "trial_2"
ALGO = "sac"
N_TRAININGS = 30

if __name__ == "__main__":
    for i in range(N_TRAININGS):
        train(
            algo=ALGO,
            indice=i,
            experiment=EXP_NAME,
            device='cuda',
            n_envs=10,
            time_steps=600_000,
            evaluate=True,
            env_kwargs={},
            seed=i,
        )
