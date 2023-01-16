from dm_control import mjcf
from dm_control import composer
from math import pi, sin, cos, tan
import numpy as np
import mujoco
import mujoco_viewer
import glfw
import imageio


class MujocoViewer(mujoco_viewer.MujocoViewer):
    def __init__(self, model, data, *args, **kwargs):
        super().__init__(model, data, *args, **kwargs)
        glfw.set_key_callback(self.window, self._key_callback)

    def _key_callback(self, window, key, scancode, action, mods):

        if action != glfw.RELEASE:
            mod_shift = (
                glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
            if key == glfw.KEY_LEFT_ALT:
                self._hide_menus = False
            elif key == glfw.KEY_UP:
                i = 2 if mod_shift else 0
                self.data.ctrl[i] = 1
            elif key == glfw.KEY_DOWN:
                i = 2 if mod_shift else 0
                self.data.ctrl[i] = -1
            elif key == glfw.KEY_LEFT:
                self.data.ctrl[1] = -1
            elif key == glfw.KEY_RIGHT:
                self.data.ctrl[1] = 1
            return
        # Switch cameras
        elif key == glfw.KEY_TAB:
            self.cam.fixedcamid += 1
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Pause simulation
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        # elif key == glfw.KEY_RIGHT and self._paused is not None:
            # self._advance_by_one_step = True
            # self._paused = True
        # Slows down simulation
        elif key == glfw.KEY_S and mods != glfw.MOD_CONTROL:
            self._run_speed /= 2.0
        # Speeds up simulation
        elif key == glfw.KEY_F:
            self._run_speed *= 2.0
        # Turn off / turn on rendering every frame.
        elif key == glfw.KEY_D:
            self._render_every_frame = not self._render_every_frame
        # Capture screenshot
        elif key == glfw.KEY_T:
            img = np.zeros(
                (glfw.get_framebuffer_size(
                    self.window)[1], glfw.get_framebuffer_size(
                    self.window)[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
            imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
            self._image_idx += 1
        # Display contact forces
        elif key == glfw.KEY_C:
            self._contacts = not self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
        elif key == glfw.KEY_J:
            self._joints = not self._joints
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = self._joints
        # Display mjtFrame
        elif key == glfw.KEY_E:
            self.vopt.frame += 1
            if self.vopt.frame == mujoco.mjtFrame.mjNFRAME.value:
                self.vopt.frame = 0
        # Hide overlay menu
        elif key == glfw.KEY_LEFT_ALT:
            self._hide_menus = True
        elif key == glfw.KEY_H:
            self._hide_menus = not self._hide_menus
        # Make transparent
        elif key == glfw.KEY_R:
            self._transparent = not self._transparent
            if self._transparent:
                self.model.geom_rgba[:, 3] /= 5.0
            else:
                self.model.geom_rgba[:, 3] *= 5.0
        # Display inertia
        elif key == glfw.KEY_I:
            self._inertias = not self._inertias
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = self._inertias
        # Display center of mass
        elif key == glfw.KEY_M:
            self._com = not self._com
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_COM] = self._com
        # Shadow Rendering
        elif key == glfw.KEY_O:
            self._shadows = not self._shadows
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self._shadows
        # Convex-Hull rendering
        elif key == glfw.KEY_V:
            self._convex_hull_rendering = not self._convex_hull_rendering
            self.vopt.flags[
                mujoco.mjtVisFlag.mjVIS_CONVEXHULL
            ] = self._convex_hull_rendering
        # Wireframe Rendering
        elif key == glfw.KEY_W:
            self._wire_frame = not self._wire_frame
            self.scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = self._wire_frame
        # Geom group visibility
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        elif key == glfw.KEY_S and mods == glfw.MOD_CONTROL:
            cam_config = {
                "type": self.cam.type,
                "fixedcamid": self.cam.fixedcamid,
                "trackbodyid": self.cam.trackbodyid,
                "lookat": self.cam.lookat.tolist(),
                "distance": self.cam.distance,
                "azimuth": self.cam.azimuth,
                "elevation": self.cam.elevation
            }
            try:
                with open(self.CONFIG_PATH, "w") as f:
                    yaml.dump(cam_config, f)
                print("Camera config saved at {}".format(self.CONFIG_PATH))
            except Exception as e:
                print(e)
        # Quit
        if key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            glfw.set_window_should_close(self.window, True)
        return


class Scene(object):

    def __init__(self,
                 xml_path: str = None,  # path to xml file
                 ):

        self.model = mjcf.RootElement(model='arena')
        self.model.compiler.set_attributes(
            angle='radian',
            meshdir='./meshes',
            autolimits=True,
        )
        self.model.option.set_attributes(
            tolerance=0,
            timestep=0.002,
            viscosity=3.5,
            density=997,
            solver='Newton',         # pgs, cg, newton
            integrator='Euler',      # euler, rk4, implicit
            cone='pyramidal',          # pyramidal, elliptic
            jacobian='sparse',
        )

        self.model.option.flag.set_attributes(
            multiccd='disable',
            frictionloss="disable",
            gravity="enable",
        )

        self.model.size.set_attributes(
            njmax=2000,
            nconmax=1000,
        )

        self._top_camera = self.model.worldbody.add(
            'camera',
            name='top_camera',
            pos=[0, 0, 0.01],
            quat=[1, 0, 0, 0],)

        self.model.asset.add('texture', type="skybox", builtin="gradient",
                             rgb1=[1, 1, 1], rgb2=[1, 1, 1],
                             width=256, height=256)
        self.model.worldbody.add('light', pos=[0, 0, 10], dir=[20, 20, -20],
                                 castshadow=False)

        if xml_path is not None:
            model = mjcf.from_file(xml_path, model_dir='./assets')
            # adjust the imported defaults
            model.default.geom.rgba = [111 / 255, 18 / 255, 0 / 255, 0]
            model.default.geom.margin = 0.004
            visual = model.find('geom', 'visual')
            visual.rgba = [111 / 255, 18 / 255, 0 / 255, 0.3]
            self.model.attach(model)

    def regenerate(self, random_state):
        pass


class Tip(object):

    def __init__(self,
                 diameter: float = 0.001,
                 n_bodies: int = 4):

        sphere_radius = (diameter / 2)
        cylinder_height = sphere_radius * 1.5
        offset = sphere_radius + cylinder_height * 2

        self.length = cylinder_height * 2 + sphere_radius * 2 + offset * n_bodies
        self.twist = False
        self.stretch = False
        self.n = 1

        self.model = mjcf.RootElement()
        # defaults
        default = self.model.default
        # geom defaults
        default.geom.rgba = [0.2, 0.2, 0.2, 1]
        default.geom.type = 'capsule'
        default.geom.size = [sphere_radius, cylinder_height]
        # joint defaults
        default.joint.type = 'hinge'
        default.joint.pos = [0, 0, -offset / 2]
        default.joint.ref = 0
        default.joint.damping = 0.5
        default.joint.stiffness = 1
        default.joint.armature = 0.05
        # actuator defaults

        parent = self.model.worldbody.add(
            'body', name='tip_0', euler=[0, 0, 0], pos=[0, 0, 0])
        parent.add('geom', name='tip_geom_0', type='capsule',
                   size=[sphere_radius, cylinder_height])
        parent.add('joint', name='T0_0', axis=[0, 0, 1])
        parent.add('joint', name='T1_0', axis=[0, 1, 0])

        # make the main body
        for i in range(n_bodies - 1):
            parent = self.add_body(parent, offset, ref=pi / 2 / n_bodies - 1)

    def add_body(self,
                 parent: mjcf.Element,  # the parent body
                 offset: float,  # the offset from the parent body
                 ref: float = None,  # the reference angle of the joint
                 ):
        child_name = f'tip_{self.n}'
        child = parent.add('body', name=child_name, pos=[0, 0, offset])
        child.add('geom', name=f'capsule_{self.n}')
        child.add('joint', name=f'J0_{self.n}', axis=[1, 0, 0])
        j1 = child.add('joint', name=f'J1_{self.n}', axis=[0, 1, 0])
        if ref is not None:
            j1.ref = ref
        if self.twist:
            child.add('joint', type='hinge',
                      name=f'JT_{self.n}', axis=[0, 0, 1])
        if self.stretch:
            child.add('joint', type='slide',
                      name=f'JS_{self.n}', axis=[0, 0, 1])

        self.n += 1

        return child


def add_body(
        n: int = 0,
        offset: float = 0,
        parent: mjcf.Element = None,  # the parent body
        ref: float = None,  # the reference angle of the joint
        stiffness: float = None,  # the stiffness of the joint
        stretch: bool = False,
        twist: bool = False,
):
    child = parent.add('body', name=f"body_{n}", pos=[0, 0, offset])
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

    def _build(self):

        diameter: float = 0.001  # the diameter of the guidewire
        n_bodies: int = 80  # number of bodies present in the guidewire
        scale: float = 1  # scaling the model
        force: float = 300  # force for translation and rotation

        sphere_radius = (diameter / 2) * scale
        cylinder_height = sphere_radius * 1.5
        offset = sphere_radius + cylinder_height * 2

        self.length = cylinder_height * 2 + sphere_radius * 2 + offset * n_bodies
        self.twist = False
        self.stretch = False
        self.n = 1
        tip_n_bodies = 4

        self.model = mjcf.RootElement()
        # defaults
        default = self.model.default
        # geom defaults
        self.model.default.geom.set_attributes(
            rgba=[0.2, 0.2, 0.2, 1],
            type='capsule',
            size=[sphere_radius, cylinder_height],
            density=7980,
        )

        self.model.default.joint.set_attributes(
            type='hinge',
            pos=[0, 0, -offset / 2],
            ref=0,
            damping=0.005,
            stiffness=2,
            springref=0,
            armature=0.05,
        )
        # actuator defaults
        self.model.default.velocity.set_attributes(
            ctrlrange=[-1, 1],
            forcerange=[-force, force],
            kv=10,
        )

        self.model.default.site.set_attributes(
            type='sphere',
            size=[sphere_radius],
        )

        parent = self.model.worldbody.add('body',
                                          name='body_0',
                                          pos=[0, -self.length, 0],
                                          euler=[-pi / 2, 0, pi],
                                          )
        parent.add('geom', name='geom_0')
        parent.add('joint', type='slide', name='slider',
                   axis=[0, 0, 1], range=[-0, 0.2])
        parent.add('joint', type='hinge', name='rotator',
                   axis=[0, 0, 1], stiffness=0, damping=2)
        self.model.actuator.add(
            'velocity', joint='slider', name='slider_actuator')
        self.model.actuator.add(
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
            parent = add_body(n, offset, parent, ref=0)

        site = parent.add('site', name='test', pos=[0, 0, offset])

        tip = Tip(diameter=diameter, n_bodies=tip_n_bodies)
        site.attach(tip.model)

    @property
    def mjcf_model(self):
        return self.model


class Catheter(object):

    def __init__(self,
                 diameter: float = 0.003,  # the diameter of the catheter
                 n_bodies: int = 40,  # number of bodies present in the catheter
                 scale: int = 1,  # scaling the model
                 force: float = 200,  # force for translation and rotation
                 ):

        offset = 0.002 + 0.0002
        length = offset * 2 * n_bodies
        radius = diameter / 2 * scale
        thickness = 0.002 * scale
        segment_length = 0.002 * scale

        self.n = 1

        self.mjcf_model = mjcf.RootElement(model='guidewire')
        default = self.mjcf_model.default
        # geom defaults
        default.geom.rgba = [0.2, 0.2, 0.2, 1]
        default.geom.type = 'box'
        default.geom.friction = [0.1, 0.1, 0.1]
        # joint defaults
        default.joint.type = 'hinge'
        default.joint.pos = [0, 0, -offset / 2]
        default.joint.ref = 0
        default.joint.damping = 0.005
        default.joint.stiffness = 0.0001
        default.joint.armature = 0.05
        # actuator defaults
        default.velocity.ctrllimited = True
        default.velocity.ctrlrange = [-1, 1]
        default.velocity.forcelimited = True
        default.velocity.forcerange = [-force, force]
        default.velocity.kv = 10

        worldbody = self.mjcf_model.worldbody
        parent = worldbody.add('body', name='geom_0', pos=[
                               0, -length, 0], euler=[-pi / 2, 0, pi],)
        self._create_body_geom(parent, 12)
        parent.add('joint', type='slide', name='catheter_slider', axis=[0, 0, 1],
                   damping=0.005, limited=True, range=[-0, 0.2])
        # parent.add('joint', type='hinge', name='rotator', axis=[0, 0, 1])
        self.mjcf_model.actuator.add('velocity', joint='catheter_slider')
        # self.mjcf_model.actuator.add('velocity', joint='rotator')

        for i in range(n_bodies - 1):
            parent = self.add_body(
                parent, offset=offset * 2, ref=0, stiffness=0.5)

    @staticmethod
    def _create_body_geom(body, n_bodies, radius=0.001, thickness=0.0002, depth=0.002):
        theta = 0
        angle_displacement = 2 * pi / n_bodies
        width = tan(angle_displacement / 2) * (radius + thickness)
        size = [thickness, width, depth]
        for n in range(n_bodies):
            pos = [radius * cos(theta), radius * sin(theta), 0]
            body.add('geom', size=size, pos=pos, euler=[0, 0, theta])
            theta += angle_displacement

    def add_body(self, parent, offset, ref=None, stiffness=None, damping=None):
        child_name = f'geom{self.n}'
        child = parent.add('body', name=child_name, pos=[0, 0, offset])
        self._create_body_geom(child, 12)
        child.add('joint', name=f'J0_{self.n}', axis=[
                  1, 0, 0], stiffness=stiffness)
        j1 = child.add('joint', name=f'J1_{self.n}', axis=[
                       0, 1, 0], stiffness=stiffness)
        if ref is not None:
            j1.ref = ref
        self.n += 1

        return child


def get_forces(model, data):
    forces = []
    for i in range(data.ncon):
        force = np.zeros(shape=6)
        mujoco.mj_contactForce(model, data, i, force)
        forces.append(force)
    return forces


if __name__ == '__main__':
    scene = Scene('./assets/phantom3.xml')

    guidewire = Guidewire()
    scene.model.attach(guidewire.model)

    mj_model = scene
    mj_model_string = mj_model.model.to_xml_string()
    assets = mj_model.model.get_assets()
    model = mujoco.MjModel.from_xml_string(mj_model_string, assets=assets)
    data = mujoco.MjData(model)
    viewer = MujocoViewer(model, data)

    # simulate and render
    for i in range(int(1 / 0.02 * 26)):
        if viewer.is_alive:
            # Collect events until released
            mujoco.mj_step(model, data)
            # data.ctrl[:] = [0.2, 0]
            viewer.render()

    # close
    viewer.close()
