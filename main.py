import mujoco
import mujoco.viewer
from robot_descriptions import panda_mj_description


world_spec = mujoco.MjSpec.from_file("world.xml")
panda_spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)

table: mujoco._specs.MjsBody = world_spec.worldbody.bodies[0]

attachment_site = table.add_site(name="robot_mount", pos=[0, -0.5, 0.03])
world_spec.attach(panda_spec, site=attachment_site)


model = world_spec.compile()
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        viewer.sync()

# try:
#     mujoco.viewer.launch(model, data)
# except KeyboardInterrupt:
#     print("ending")
