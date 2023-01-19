from cathsim import Phantom, Tip, Guidewire, Navigate
import mujoco
from mujoco import viewer
phantom = Phantom("assets/phantom4.xml", model_dir="./assets")
tip = Tip(n_bodies=4)
guidewire = Guidewire(n_bodies=80)
task = Navigate(
    phantom=phantom,
    guidewire=guidewire,
    tip=tip,
    use_image=True
)
model_string = task._arena._mjcf_root.to_xml_string(precision=3)
model_assets = task._arena._mjcf_root.get_assets()
model = mujoco.MjModel.from_xml_string(model_string, model_assets)

mujoco.viewer.launch(model)
#
# task._arena._mjcf_root.to_xml_string(precision=3)
# with open("model.xml", "w") as text_file:
#     text_file.write(model_string)
# exit()
