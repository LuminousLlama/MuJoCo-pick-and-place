import mujoco
import numpy as np
from robot_descriptions import panda_mj_description


def create_env() -> mujoco.MjModel:
    world_spec = mujoco.MjSpec()
    world = world_spec.worldbody

    light = world.add_light()
    light.pos = np.array([0, 0, 3])

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

    # Add table legs (4 cylinders)
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

    panda_spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
    attachment_site = table.add_site(name="robot_mount", pos=[0, -0.65, 0.03])
    world_spec.attach(child=panda_spec, site=attachment_site)

    # cube to pick up
    cube = world.add_body()
    cube.name = "cube"
    cube.pos = np.array([0, 0, 2])
    cube.add_freejoint()

    cube_geom = cube.add_geom()
    cube_geom.name = "cube_geom"
    cube_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    cube_geom.size = np.array([0.02, 0.02, 0.02])
    cube_geom.rgba = np.array([0.8, 0, 0, 1])
    cube_geom.mass = 0.2

    return world_spec.compile()
