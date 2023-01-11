import math
import cv2
import numpy as np

from dm_control import mjcf, mujoco, composer
from dm_control.composer import variation
from dm_control.composer.variation import distributions, noises
from dm_control.composer.observation import observable
from dm_control.suite.wrappers.pixels import Wrapper
from dm_control.composer.observation.observable import MujocoCamera, MujocoFeature


_DEFAULT_TIME_LIMIT = 50
_CONTROL_TIMESTEP = .005

SCALE = 1
RGBA = [0.2, 0.2, 0.2, 1]
BODY_DIAMETER = 0.001
SPHERE_RADIUS = (BODY_DIAMETER / 2) * SCALE
CYLINDER_HEIGHT = SPHERE_RADIUS * 1.5
OFFSET = SPHERE_RADIUS + CYLINDER_HEIGHT * 2
TWIST = False
STRETCH = False
FORCE = 300
STIFFNESS = 10
TARGET_POS = (-0.043094, 0.14015, 0.033013)
MARGIN = 0.004
CONDIM = 3
FRICTION = [0.2, 0.002, 0.001]

NUM_SUBSTEPS = 4

TIP_N_BODIES = 3
TIP_REF = math.pi / 2 / TIP_N_BODIES - 1

random_state = np.random.RandomState(42)


class Scene(composer.Arena):

    def _build(self,
               name: str = 'arena',
               render_site: bool = True,
               ):
        super()._build(name=name)

        self._mjcf_root.compiler.set_attributes(
            angle='radian',
            meshdir='./meshes',
            autolimits=True,
        )
        self._mjcf_root.option.set_attributes(
            timestep=_CONTROL_TIMESTEP,
            viscosity=0.0009 * 4,
            density=1060,
            solver='newton',         # pgs, cg, newton
            integrator='euler',      # euler, rk4, implicit
            cone='pyramidal',          # pyramidal, elliptic
            # jacobian='sparse',
        )

        self._mjcf_root.option.flag.set_attributes(
            # multiccd='disable',
            # frictionloss="disable",
            gravity="disable",
        )

        self._mjcf_root.size.set_attributes(
            njmax=2000,
            nconmax=2000,
            # memory="1G",
        )

        self._mjcf_root.default.site.set_attributes(
            type='sphere',
            size=[0.004],
            rgba=[0.8, 0.8, 0.8, 0],
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

        site = self._mjcf_root.worldbody.add('site', pos=TARGET_POS)
        if render_site:
            site.rgba = self._mjcf_root.default.site.rgba
            site.rgba[-1] = 1

    def regenerate(self, random_state):
        pass


class CameraObservable(MujocoCamera):
    def __init__(self, camera_name, height=128, width=128, corruptor=None,
                 depth=False, preprocess=True, grayscale=True):
        super().__init__(camera_name, height, width, corruptor, depth)
        self._preprocess = preprocess
        self._grayscale = grayscale
        self._dtype = np.float32 if depth or grayscale or preprocess else np.int8
        self._n_channels = 1 if depth or grayscale else 3

    def _callable(self, physics):
        def get_image():
            image = physics.render(  # pylint: disable=g-long-lambda
                self._height, self._width, self._camera_name, depth=self._depth)
            if self._grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self._preprocess and not self._depth:
                image = image / 255.0 - 0.5
            return image

        return get_image


def add_body(
        n: int = 0,
        parent: mjcf.Element = None,  # the parent body
        geom_size: float = None,
        ref: float = None,  # the reference angle of the joint
        stiffness: float = None,  # the stiffness of the joint
        stretch: bool = False,
        twist: bool = False,
):
    child = parent.add('body', name=f"body_{n}")
    child.add('geom', name=f'geom_{n}')
    child.pos = [0, 0, geom_size[0] + geom_size[1] * 2]
    j0 = child.add('joint', name=f'J0_{n}', axis=[1, 0, 0])
    j1 = child.add('joint', name=f'J1_{n}', axis=[0, 1, 0])
    if stiffness is not None:
        j0.stiffness = stiffness
        j1.stiffness = stiffness

    return child


class Guidewire(composer.Entity):

    def _build(self, n_bodies: int = 80):

        self._length = CYLINDER_HEIGHT * 2 + SPHERE_RADIUS * 2 + OFFSET * n_bodies

        self._mjcf_root = mjcf.RootElement(model="guidewire")

        self._mjcf_root.default.geom.set_attributes(
            group=1,
            rgba=[0.1, 0.1, 0.1, 1],
            type='capsule',
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            density=7980,
            margin=MARGIN,
            condim=CONDIM,
            friction=FRICTION,
            # fluidshape='ellipsoid',
        )

        self._mjcf_root.default.joint.set_attributes(
            type='hinge',
            pos=[0, 0, -OFFSET / 2],
            ref=0,
            damping=0.005,
            stiffness=STIFFNESS,
            springref=0,
            armature=0.05,
            axis=[0, 0, 1],
        )

        self._mjcf_root.default.site.set_attributes(
            type='sphere',
            size=[SPHERE_RADIUS],
            rgba=[0.3, 0, 0, 0.0],
        )

        self._mjcf_root.default.velocity.set_attributes(
            ctrlrange=[-1, 1],
            forcerange=[-FORCE, FORCE],
            kv=10,
        )

        parent = self._mjcf_root.worldbody.add('body',
                                               name='body_0',
                                               euler=[-math.pi / 2, 0, math.pi],
                                               )
        parent.add('geom', name='geom_0')
        parent.add('joint', type='slide', name='slider', range=[-0, 0.2])
        parent.add('joint', type='hinge', name='rotator', stiffness=0, damping=2)
        self._mjcf_root.actuator.add('velocity', joint='slider', name='slider_actuator')
        self._mjcf_root.actuator.add('general', joint='rotator', name='rotator_actuator',
                                     dyntype=None, gaintype='fixed', biastype='None',
                                     dynprm=[1, 0, 0], gainprm=[40, 0, 0], biasprm=[2])

        # make the main body
        stiffness = self._mjcf_root.default.joint.stiffness
        for n in range(1, n_bodies):
            parent = add_body(n, parent, stiffness=stiffness, geom_size=self._mjcf_root.default.geom.size)
            stiffness *= 0.98

        self._tip_site = parent.add('site', name='tip_site', pos=[0, 0, OFFSET])

    @property
    def attachment_site(self):
        return self._tip_site

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def actuators(self):
        return tuple(self._mjcf_root.find_all('actuator'))

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all('joint'))

    def _build_observables(self):
        return GuidewireObservables(self)


class GuidewireObservables(composer.Observables):

    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)

    @composer.observable
    def actuators_control(self):
        all_actuators = self._entity.mjcf_model.find_all('actuator')
        return observable.MJCFFeature('ctrl', all_actuators)


class Phantom(composer.Entity):
    def _build(self, xml_path, **kwargs):
        self._rgba = [111 / 255, 18 / 255, 0 / 255, 0]
        self._mjcf_root = mjcf.from_file(xml_path, **kwargs)
        self._mjcf_root.model = "phantom"
        self._mjcf_root.default.geom.set_attributes(
            group=0,
            rgba=self._rgba,
            margin=MARGIN,
            condim=CONDIM,
            friction=FRICTION,
        )
        self._rgba[-1] = 0.3
        self._mjcf_root.find('geom', 'visual').rgba = self._rgba

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def _build_observables(self):
        return PhantomObservables(self)


class PhantomObservables(composer.Observables):

    @composer.observable
    def geom_pos(self):
        all_geoms = self._entity.mjcf_model.find_all('geom')
        return observable.MJCFFeature('pos', all_geoms)


class Tip(composer.Entity):

    def _build(self, name=None, n_bodies=3):

        if name is None:
            name = 'tip'
        self._mjcf_root = mjcf.RootElement(model=name)

        self._mjcf_root.default.geom.set_attributes(
            group=2,
            rgba=[0., 0.2, 0, 1],
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            type="capsule",
            margin=MARGIN,
            condim=CONDIM, friction=FRICTION,
        )

        self._mjcf_root.default.joint.set_attributes(
            type='hinge',
            pos=[0, 0, -OFFSET / 2],
            springref=math.pi / 4 / n_bodies,
            # ref=pi / 2 / n_bodies - 1,
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
            parent = add_body(n, parent, geom_size=self._mjcf_root.default.geom.size)

        self.head_geom.name = 'head'

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all('joint'))

    @property
    def head_geom(self):
        return self._mjcf_root.find_all('geom')[-1]

    def _build_observables(self):
        return TipObservables(self)


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
                 delta: float = 0.004,  # distance threshold for success
                 dense_reward: bool = True,
                 success_reward: float = 10.0,
                 use_image: bool = False,
                 use_action: bool = True,
                 ):

        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward
        self.use_image = use_image
        self.use_action = use_action

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
        self._guidewire_initial_pose = [0, -(self._guidewire._length - 0.015), 0]
        self._target_pos = TARGET_POS

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
        pos_corrptor = noises.Additive(distributions.Normal(scale=0.0001))
        self._guidewire.observables.joint_positions.corruptor = pos_corrptor
        vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.0001))
        self._guidewire.observables.joint_velocities.corruptor = vel_corruptor

        self._guidewire.observables.joint_positions.enabled = True
        self._guidewire.observables.joint_velocities.enabled = True
        self._guidewire.observables.actuators_control.enabled = True

        self._tip.observables.joint_positions.enabled = True
        self._tip.observables.joint_velocities.enabled = True

        self._phantom.observables.geom_pos.enabled = True

        self._task_observables = {}

        if self.use_image:
            self._task_observables['top_camera'] = CameraObservable(
                camera_name='top_camera',
                width=128,
                height=128,
            )

        # self._task_observables['contact_force'] = observable.Generic(self.get_contact_forces)
        # self._task_observables['contact_pos'] = observable.Generic(self.get_contact_positions)
        self._task_observables['activation'] = observable.Generic(self.get_control)

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
        guidewire_pose = variation.evaluate(self._guidewire_initial_pose,
                                            random_state=random_state)
        self._guidewire.set_pose(physics, position=guidewire_pose)
        self.success = False

    def get_reward(self, physics):
        head_pos = physics.named.data.geom_xpos[-1]
        reward = self.compute_reward(head_pos, self._target_pos)
        return reward

    def should_terminate_episode(self, physics):
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

    def get_control(self, physics):
        return physics.control()

    def get_contact_forces(self, physics):
        contact_forces = []
        for i in range(physics.data.ncon):
            contact_force = physics.data.contact_force(i)
            contact_forces.append(contact_force)
        return contact_forces

    def get_contact_positions(self, physics):
        return physics.data.contact.pos


class Physics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    def joint_pos(self):
        pass

    def joint_vel(self):
        pass

    def site_distance(self, site1, site2):
        site1_to_site2 = np.diff(self.named.data.site_xpos[[site2, site1]], axis=0)
        return np.linalg.norm(site1_to_site2)

    def get_contacts(self):
        self.data.contact_force
        return

    def get_activation(self):
        return self.activation()


if __name__ == "__main__":
    from dm_control import viewer

    phantom = Phantom("assets/phantom3.xml", model_dir="./assets")
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    # print(Guidewire.actuators[0])
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_image=True
    )
    env = composer.Environment(
        task=task,
        time_limit=_DEFAULT_TIME_LIMIT,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    mjcf.export_with_assets(task._arena._mjcf_root, './model')

    action_spec = env.action_spec()
    time_step = env.reset()
    for key, value in time_step.observation.items():
        print(key, value.shape)

    def random_policy(time_step):
        del time_step
        return np.random.uniform(low=action_spec.minimum,
                                 high=action_spec.maximum,
                                 size=action_spec.shape)

    viewer.launch(env, policy=random_policy)
