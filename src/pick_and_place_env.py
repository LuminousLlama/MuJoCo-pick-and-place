import mujoco
import numpy as np
from robot_descriptions import panda_mj_description


def create_env() -> mujoco.MjModel:
    world_spec = mujoco.MjSpec()
    world = world_spec.worldbody

    light = world.add_light()
    light.pos = np.array([0, 0, 3])
    light.dir = np.array([0, 0, -1])
    light.diffuse = np.array([0.7, 0.7, 0.7])
    light.ambient = np.array([0.4, 0.4, 0.4])
    light.specular = np.array([0.3, 0.3, 0.3])
    light.castshadow = True

    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([5, 5, 0.01])

    # table
    table = world.add_body()
    table.name = "table"
    table.pos = np.array([0, 0, 0.8])

    tabletop = table.add_geom()
    tabletop.name = "tabletop"
    tabletop.type = mujoco.mjtGeom.mjGEOM_BOX
    tabletop.size = np.array([1.0, 0.8, 0.025])
    tabletop.rgba = np.array([0.6, 0.4, 0.2, 1])

    # table legs (4 cylinders)
    leg1 = table.add_geom()
    leg1.name = "leg1"
    leg1.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    leg1.size = np.array([0.02, 0.4, 0])
    leg1.pos = np.array([0.9, 0.7, -0.4])
    leg1.rgba = np.array([0.5, 0.3, 0.1, 1])

    leg2 = table.add_geom()
    leg2.name = "leg2"
    leg2.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    leg2.size = np.array([0.02, 0.4, 0])
    leg2.pos = np.array([0.9, -0.7, -0.4])
    leg2.rgba = np.array([0.5, 0.3, 0.1, 1])

    leg3 = table.add_geom()
    leg3.name = "leg3"
    leg3.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    leg3.size = np.array([0.02, 0.4, 0])
    leg3.pos = np.array([-0.9, 0.7, -0.4])
    leg3.rgba = np.array([0.5, 0.3, 0.1, 1])

    leg4 = table.add_geom()
    leg4.name = "leg4"
    leg4.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    leg4.size = np.array([0.02, 0.4, 0])
    leg4.pos = np.array([-0.9, -0.7, -0.4])
    leg4.rgba = np.array([0.5, 0.3, 0.1, 1])

    # goal zone
    goal = table.add_geom()
    goal.name = "goal"
    goal.type = mujoco.mjtGeom.mjGEOM_BOX
    goal.size = np.array([0.15, 0.15, 0.01])
    goal.pos = np.array([-0.9 + 0.15, -0.7 + 0.15, 0.03])
    goal.rgba = np.array([0.0, 0.5, 0.0, 0.9])

    panda_spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
    # use our own light instead
    panda_spec.delete(panda_spec.lights[0])

    attachment_site = table.add_site(
        name="robot_mount",
        pos=[-0.8, 0, 0.03],
    )
    world_spec.attach(child=panda_spec, site=attachment_site)

    # objects on table
    cube_0 = world.add_body()
    cube_0.name = "cube_0"
    cube_0.pos = np.array([-0.3, 0, 1])
    cube_0.add_freejoint()

    cube_geom_0 = cube_0.add_geom()
    cube_geom_0.name = "cube_geom_0"
    cube_geom_0.type = mujoco.mjtGeom.mjGEOM_BOX
    cube_geom_0.size = np.array([0.05, 0.02, 0.02])
    cube_geom_0.rgba = np.array([0.8, 0, 0, 1])
    cube_geom_0.mass = 0.2
    # cube_geom.friction = np.array([3.0, 0.005, 0.0001])
    cube_geom_0.solref = np.array([0.01, 1])
    cube_geom_0.solimp = np.array([0.9, 0.99, 0.001, 0.5 * 0.001, 2])
    cube_geom_0.condim = 6

    cube_1 = world.add_body()
    cube_1.name = "cube_1"
    cube_1.pos = np.array([-0.3, 0.4, 1])
    cube_1.add_freejoint()

    cube_geom_1 = cube_1.add_geom()
    cube_geom_1.name = "cube_geom_1"
    cube_geom_1.type = mujoco.mjtGeom.mjGEOM_BOX
    cube_geom_1.size = np.array([0.02, 0.02, 0.02])
    cube_geom_1.rgba = np.array([0, 0, 0.8, 1])
    cube_geom_1.mass = 0.2
    cube_geom_1.solref = np.array([0.01, 1])
    cube_geom_1.solimp = np.array([0.9, 0.99, 0.001, 0.5 * 0.001, 2])
    cube_geom_1.condim = 6

    cylinder = world.add_body()
    cylinder.name = "cylinder"
    cylinder.name = "cylinder"
    cylinder.pos = np.array([-0.4, -0.5, 1])
    cylinder.add_freejoint()

    cylinder_geom = cylinder.add_geom()
    cylinder_geom.name = "cylinder_geom"
    cylinder_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    cylinder_geom.size = np.array([0.02, 0.04, 0])
    cylinder_geom.rgba = np.array([0, 0.8, 0, 1])
    cylinder_geom.mass = 0.2

    cylinder_geom.friction = np.array([1.0, 0.005, 0.0004])
    cylinder_geom.solref = np.array([0.01, 1])
    cylinder_geom.solimp = np.array([0.9, 0.99, 0.001, 0.5 * 0.001, 2])
    cylinder_geom.condim = 6

    return world_spec.compile()
