from utils import train

EXP_NAME = "multi_input"
ALGO = "sac"
policy = 'MultiInputPolicy'
N_TRAININGS = 10

if __name__ == "__main__":
    wrapper_kwargs = dict(
        time_limit=300,
        use_pixels=True,
        grayscale=True,
        resize_shape=80,
    )

    algo_kwargs = dict(
        policy='MultiInputPolicy',
    )

    env_kwargs = dict(
    )

    for i in range(N_TRAININGS):
        train(
            algo=ALGO,
            indice=i,
            experiment=EXP_NAME,
            device='cuda',
            n_envs=8,
            time_steps=500_000,
            evaluate=True,
            env_kwargs=env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
            algo_kwargs=algo_kwargs,
            seed=i,
        )
