from utils import train
from rl.models.vit_policy import CombinedExtractor

EXP_NAME = "testing_simple"
ALGO = "sac"
policy = 'MultiInputPolicy'
N_TRAININGS = 6

if __name__ == "__main__":
    task_kwargs = dict(
        # use_pixels=True,
        # use_segment=True,
        # image_size=80,
    )

    wrapper_kwargs = dict(
        time_limit=300,
        # grayscale=True,
        # use_obs=[
        #     'joint_pos',
        #     'joint_vel',
        #     'pixels',
        # ],
    )

    algo_kwargs = dict(
        policy='MultiInputPolicy',
        # policy_kwargs=dict(
        #     features_extractor_class=CombinedExtractor,
        #     features_extractor_kwargs=dict(
        #         features_dim=256,
        #         image_size=task_kwargs['image_size'],
        #         patch_size=task_kwargs['image_size'] // 16,
        #     ),
        # )
    )

    env_kwargs = dict(
    )

    for i in range(N_TRAININGS):
        train(
            algo=ALGO,
            indice=i,
            experiment=EXP_NAME,
            device='cuda',
            # n_envs=8,
            time_steps=500_000,
            evaluate=True,
            env_kwargs=env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
            algo_kwargs=algo_kwargs,
            task_kwargs=task_kwargs,
            seed=i,
        )
