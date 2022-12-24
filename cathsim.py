from dm_control import mjcf
from pprint import pprint
from dm_control.composer.observation import observable
from math import pi, sin, cos, tan
import numpy as np
import mujoco
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises

SCALE = 1
RGBA = [0.2, 0.2, 0.2, 1]
BODY_DIAMETER = 0.001
SPHERE_RADIUS = (BODY_DIAMETER / 2) * SCALE
CYLINDER_HEIGHT = SPHERE_RADIUS * 1.5
OFFSET = SPHERE_RADIUS + CYLINDER_HEIGHT * 2
TWIST = False
STRETCH = False
FORCE = 300
TARGET_POS = (-0.043094, 0.13715, 0.033513)

NUM_SUBSTEPS = 5

TIP_N_BODIES = 4
TIP_REF = pi / 2 / TIP_N_BODIES - 1

random_state = np.random.RandomState(42)


class Scene(composer.Arena):

    def _build(self, name: str = 'arena'):
        super()._build(name=name)

        self._mjcf_root.compiler.set_attributes(
            angle='radian',
            meshdir='./meshes',
            autolimits=True,
        )
        self._mjcf_root.option.set_attributes(
            tolerance=0,
            timestep=0.002,
            viscosity=3.5,
            density=997,
            solver='Newton',         # pgs, cg, newton
            integrator='Euler',      # euler, rk4, implicit
            cone='pyramidal',          # pyramidal, elliptic
            jacobian='sparse',
        )

        self._mjcf_root.option.flag.set_attributes(
            multiccd='disable',
            frictionloss="disable",
            gravity="enable",
        )

        self._mjcf_root.size.set_attributes(
            njmax=2000,
            nconmax=1000,
        )

        self._mjcf_root.default.site.set_attributes(
            type='sphere',
            size=[SPHERE_RADIUS],
            rgba=[0.3, 0, 0, 0.7],
        )

        self._top_camera = self._mjcf_root.worldbody.add(
            'camera',
            name='top_camera',
            pos=[-0.03, 0.09, 0.230],
            euler=[0, 0, 0],)

        self._mjcf_root.asset.add('texture', type="skybox", builtin="gradient",
                                  rgb1=[1, 1, 1], rgb2=[1, 1, 1],
                                  width=256, height=256)
        self._mjcf_root.worldbody.add('light', pos=[0, 0, 10], dir=[20, 20, -20],
                                      castshadow=False)

        self._mjcf_root.worldbody.add('site', pos=TARGET_POS)

    def regenerate(self, random_state):
        pass


def add_body(
        n: int = 0,
        parent: mjcf.Element = None,  # the parent body
        ref: float = None,  # the reference angle of the joint
        stiffness: float = None,  # the stiffness of the joint
        stretch: bool = False,
        twist: bool = False,
):
    child = parent.add('body', name=f"body_{n}", pos=[0, 0, OFFSET])
    child.add('geom', name=f'geom_{n}')
    j0 = child.add('joint', name=f'J0_{n}', axis=[1, 0, 0])
    j1 = child.add('joint', name=f'J1_{n}', axis=[0, 1, 0])
    if ref is not None:
        j1.ref = ref
    if stiffness is not None:
        j1.stiffness = stiffness
        j0.stiffness = stiffness

    return child


class Guidewire(composer.Entity):

    def _build(self, n_bodies: int = 80):

        self._length = CYLINDER_HEIGHT * 2 + SPHERE_RADIUS * 2 + OFFSET * n_bodies

        self._mjcf_root = mjcf.RootElement(model="guidewire")

        self._mjcf_root.default.geom.set_attributes(
            rgba=[0.2, 0.2, 0.2, 1],
            type='capsule',
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            density=7980,
        )

        self._mjcf_root.default.joint.set_attributes(
            type='hinge',
            pos=[0, 0, -OFFSET / 2],
            ref=0,
            damping=0.005,
            stiffness=2,
            springref=0,
            armature=0.05,
        )

        self._mjcf_root.default.site.set_attributes(
            type='sphere',
            size=[SPHERE_RADIUS],
            rgba=[0.3, 0, 0, 0.7],
        )

        self._mjcf_root.default.velocity.set_attributes(
            ctrlrange=[-1, 1],
            forcerange=[-FORCE, FORCE],
            kv=10,
        )

        parent = self._mjcf_root.worldbody.add('body',
                                               name='body_0',
                                               pos=[0, -self._length, 0],
                                               euler=[-pi / 2, 0, pi],
                                               )
        parent.add('geom', name='geom_0')
        parent.add('joint', type='slide', name='slider', axis=[0, 0, 1], range=[-0, 0.2])
        parent.add('joint', type='hinge', name='rotator', axis=[0, 0, 1], stiffness=0, damping=2)
        self._mjcf_root.actuator.add('velocity', joint='slider', name='slider_actuator')
        self._mjcf_root.actuator.add(
            'general',
            joint='rotator',
            name='rotator_actuator',
            dyntype=None,
            gaintype='fixed',
            biastype='None',
            dynprm=[1, 0, 0],
            gainprm=[40, 0, 0],
            biasprm=[2],
        )

        # make the main body
        for n in range(1, n_bodies):
            parent = add_body(n, parent, ref=0)

        self._tip_site = parent.add('site', name='tip_site', pos=[0, 0, OFFSET / 2])

    @property
    def attachment_site(self):
        return self._tip_site

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def tip_site(self):
        return self._tip_site

    def _build_observables(self):
        return GuidewireObservables(self)

    @property
    def actuators(self):
        return tuple(self._mjcf_root.find_all('actuator'))

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all('joint'))


class GuidewireObservables(composer.Observables):

    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class Phantom(composer.Entity):
    def _build(self, xml_path, **kwargs):
        self._rgba = [111 / 255, 18 / 255, 0 / 255, 0]
        self._mjcf_root = mjcf.from_file(xml_path, **kwargs)
        self._mjcf_root.default.geom.set_attributes(
            rgba=self._rgba,
            margin=0.002,
        )
        self._rgba[-1] = 0.3
        self._mjcf_root.find('geom', 'visual').rgba = self._rgba

    @property
    def mjcf_model(self):
        return self._mjcf_root


class Tip(composer.Entity):

    def _build(self, name=None, n_bodies=4):

        if name is None:
            name = 'tip'
        self._mjcf_root = mjcf.RootElement(model=name)

        self._mjcf_root.default.geom.set_attributes(
            rgba=[0., 0.2, 0, 1],
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            type="capsule",
        )

        self._mjcf_root.default.joint.set_attributes(
            type='hinge',
            pos=[0, 0, -OFFSET / 2],
            ref=0.02,
            damping=0.5,
            stiffness=1,
            armature=0.05,
        )

        parent = self._mjcf_root.worldbody.add(
            'body',
            name='body_0',
            euler=[0, 0, 0],
            pos=[0, 0, 0],
        )

        parent.add('geom', name='geom_0',)
        parent.add('joint', name='T0_0', axis=[0, 0, 1])
        parent.add('joint', name='T1_0', axis=[0, 1, 0])

        for n in range(1, n_bodies):
            parent = add_body(n, parent, ref=pi / 2 / n_bodies - 1)

        self.head_geom.name = 'head'

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all('joint'))

    def _build_observables(self):
        return TipObservables(self)

    @property
    def head_geom(self):
        return self._mjcf_root.find_all('geom')[-1]


class TipObservables(composer.Observables):

    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class Navigate(composer.Task):

    def __init__(self,
                 phantom: composer.Entity = None,
                 guidewire: composer.Entity = None,
                 tip: composer.Entity = None,
                 obs_type: str = 'full',
                 delta: float = 0.008,  # distance threshold for success
                 dense_reward: bool = True,
                 success_reward: float = 10.0,
                 ):

        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward

        self._arena = Scene("arena")
        if phantom is not None:
            self._phantom = phantom
            self._arena.attach(self._phantom)
        if guidewire is not None:
            self._guidewire = guidewire
            if tip is not None:
                self._tip = tip
                self._guidewire.attach(self._tip)
            self._arena.attach(self._guidewire)

        # Configure initial poses
        self._guidewire_initial_pose = (0, 0, 0)
        self._target_pos = TARGET_POS

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

        self._tip.observables.joint_positions.enabled = True
        self._tip.observables.joint_velocities.enabled = True

        self._task_observables = {}

        for obs in self._task_observables.values():
            print(obs)
            obs.enabled = True

        self.control_timestep = NUM_SUBSTEPS * self.physics_timestep

        self.success = False

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
        self.success = False

    def get_reward(self, physics):
        head_pos = physics.named.data.geom_xpos[-1]
        reward = self.compute_reward(head_pos, self._target_pos)
        return reward

    def should_terminate_episode(self, physics):  # pylint: disable=unused-argument
        """Determines whether the episode should terminate given the physics state.

        Args:
          physics: A Physics object

        Returns:
          A boolean
        """
        return self.success

    @property
    def head_pos(self, physics):
        return physics.named.data.geom_xpos[-1]

    def compute_reward(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        success = bool(distance <= self.delta)

        if self.dense_reward:
            reward = self.success_reward if success else -distance
        else:
            reward = self.success_reward if success else -1.0
        self.success = success
        return reward
