import pandas as pd
from pprint import pprint
from math import pi, sin, cos, tan
import numpy as np
from dm_control import viewer
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from cathsim_elements import Guidewire, Scene, Phantom, Tip

SCALE = 1
RGBA = [0.2, 0.2, 0.2, 1]
BODY_DIAMETER = 0.001
SPHERE_RADIUS = (BODY_DIAMETER / 2) * SCALE
CYLINDER_HEIGHT = SPHERE_RADIUS * 1.5
OFFSET = SPHERE_RADIUS + CYLINDER_HEIGHT * 2
SIZE = [SPHERE_RADIUS, CYLINDER_HEIGHT],
TWIST = False
STRETCH = False
FORCE = 300


TIP_N_BODIES = 4
TIP_REF = pi / 2 / TIP_N_BODIES - 1

random_state = np.random.RandomState(42)


class Navigate(composer.Task):

    def __init__(self,
                 phantom: composer.Entity = None,
                 guidewire: composer.Entity = None,
                 num_substeps: int = 10,
                 ):

        self._arena = Scene("arena")
        if phantom is not None:
            self._phantom = phantom
            self._arena.attach(self._phantom)
        if guidewire is not None:
            self._guidewire = guidewire
            self._arena.attach(self._guidewire)

        # Configure initial poses
        self._guidewire_initial_pose = (0, 0, 0)
        self._target_pos = (-0.043094, 0.13715, 0.033513)

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
        pos_corrptor = noises.Additive(distributions.Normal(scale=0.0))
        self._guidewire.observables.joint_positions.corruptor = pos_corrptor
        self._guidewire.observables.joint_positions.enabled = True
        vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.0))
        self._guidewire.observables.joint_velocities.corruptor = vel_corruptor
        self._guidewire.observables.joint_velocities.enabled = True

        self._task_observables = {}

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = num_substeps * self.physics_timestep

    @property
    def to_target(self, physics):

        return self._head_pos.global_vector_to_local_frame(physics,
                                                           self._target_pos)

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        creature_pose = variation.evaluate(self._guidewire_initial_pose,
                                           random_state=random_state)
        self._guidewire.set_pose(physics, position=creature_pose)

    def get_reward(self, physics):
        head_pos = self._guidewire.head_pos
        distance = np.linalg.norm(head_pos - self._target_pos)

        return - distance


if __name__ == '__main__':
    phantom = Phantom("assets/phantom3.xml", model_dir="./assets")
    tip = Tip()
    guidewire = Guidewire(tip=tip)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )
    action_spec = env.action_spec()
    time_step = env.reset()

    def random_policy(time_step):
        print(time_step.reward)  # Unused.
        return np.random.uniform(low=action_spec.minimum,
                                 high=action_spec.maximum,
                                 size=action_spec.shape)

    # Launch the viewer application.
    viewer.launch(env, policy=random_policy)
