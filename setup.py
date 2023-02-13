from setuptools import setup, find_packages

extra_dev = [
    'opencv-python',
    'matplotlib',
]

extra_rl = [
    'torch',
    'stable-baselines3',
    'sb3_contrib',
    'imitation',
    'tqdm',
    'rich',
]

extra = extra_dev + extra_rl


setup(
    name='cathsim',
    version='dev',
    url='git@github.com:tudorjnu/packaging_test.git',
    author='Author',
    author_email='my_email',
    packages=find_packages(
        exclude=[
            'tests',
            'scripts',
            'notebooks',
            'figures',
        ]
    ),
    install_requires=[
        'dm_control',
        'gym==0.21.*',
    ],
    extras_require={
        'dev': extra_dev,
        'rl': extra_rl,
        'all': extra,
    },
    entry_points={
        'console_scripts': [
            'run_env=cathsim.utils:run_env',
            'record_traj=rl.sb3.utils:cmd_record_traj',
        ],
    },
)
