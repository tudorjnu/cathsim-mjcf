from dm_control import composer
from dm_control import viewer
import cv2
import copy
import numpy as np
from cathsim import Navigate, Tip, Guidewire, Phantom
from dm_control.utils import containers
from ray.rllib.env.wrappers.dm_env_wrapper import DMEnv

phantom = Phantom("assets/phantom3.xml", model_dir="./assets")
tip = Tip(n_bodies=4)
guidewire = Guidewire(n_bodies=80)

task = Navigate(
    phantom=phantom,
    guidewire=guidewire,
    tip=tip,
)


env = composer.Environment(
    task=task,
    time_limit=2000,
    random_state=np.random.RandomState(42),
    strip_singleton_obs_buffer_dim=True,
)

wrapped_env = DMEnv(env)
exit()

SUITE = containers.TaggedTasks()
print(SUITE.tags())


viewer.launch(env)
random_state = np.random.RandomState(42)
duration = 2
spec = env.action_spec()
time_step = env.reset()

frames = []
ticks = []
rewards = []
observations = []
while env.physics.data.time < duration:

    action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
    time_step = env.step(action)

    camera0 = env.physics.render(camera_id=0, height=200, width=200)
    camera0 = cv2.cvtColor(camera0, cv2.COLOR_BGR2RGB)
    cv2.imshow("Camera 0", camera0)
    cv2.waitKey(1)

    rewards.append(time_step.reward)
    observations.append(copy.deepcopy(time_step.observation))
    print(time_step.reward)
    ticks.append(env.physics.data.time)
