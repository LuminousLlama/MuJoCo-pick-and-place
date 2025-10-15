import mujoco
import mujoco.viewer
import pick_and_place_env
from pynput import keyboard
from sys import exit
import numpy as np

active_keys = set()

linear_speed = 0.3
angular_speed = 1.0
damping = 0.01


def on_press(key):
    try:
        active_keys.add(key.char)
    except:
        print("bad key")


def on_release(key):
    try:
        active_keys.discard(key.char)
    except:
        print("error")
        exit(1)


def update_desired_vels_from_keyboard():
    linear_vels = [0.0, 0.0, 0.0]
    angular_vels = [0.0, 0.0, 0.0]

    if "w" in active_keys:
        linear_vels[0] += linear_speed
    if "s" in active_keys:
        linear_vels[0] -= linear_speed
    if "a" in active_keys:
        linear_vels[1] += linear_speed
    if "d" in active_keys:
        linear_vels[1] -= linear_speed
    if "q" in active_keys:
        linear_vels[2] += linear_speed
    if "e" in active_keys:
        linear_vels[2] -= linear_speed

    # Rotation (Angular velocity)
    # pitch is differently flipped from others so that "i" makes it go up not down)
    if "i" in active_keys:
        angular_vels[1] -= angular_speed  # Pitch up (rotate around Y-)
    if "k" in active_keys:
        angular_vels[1] += angular_speed  # Pitch down (rotate around Y+)
    if "j" in active_keys:
        angular_vels[2] += angular_speed  # Yaw left (rotate around Z+)
    if "l" in active_keys:
        angular_vels[2] -= angular_speed  # Yaw right (rotate around Z-)
    if "u" in active_keys:
        angular_vels[0] += angular_speed  # Roll left (rotate around X+)
    if "o" in active_keys:
        angular_vels[0] -= angular_speed  # Roll right (rotate around X-)

    return linear_vels, angular_vels


def cartesian_velocity_to_joint_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    linear_velocity,
    angular_velocity,
    end_effector_body_name,
    damping=0.01,
):
    ee_id = model.body(end_effector_body_name).id
    ee_pos = data.body(ee_id).xpos.copy()

    jacp = np.zeros((3, model.nv))  # Translational Jacobian
    jacr = np.zeros((3, model.nv))  # Rotational Jacobian

    mujoco.mj_jac(model, data, jacp, jacr, ee_pos, ee_id)

    # Use only arm joints (first 7 columns)
    jacp_arm = jacp[:, :7]
    jacr_arm = jacr[:, :7]

    # Stack into full 6-DOF Jacobian [6x7]
    J_full = np.vstack([jacp_arm, jacr_arm])
    desired_twist = np.concatenate([linear_velocity, angular_velocity])

    # Damped pseudoinverse: J^T (JJ^T + λI)^-1
    product = J_full @ J_full.T + damping * np.eye(6)
    j_inv = J_full.T @ np.linalg.inv(product)

    joint_velocities = j_inv @ desired_twist

    return joint_velocities


def view_keycallbacl(keycode):
    return


def main():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    model = pick_and_place_env.create_env()
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=view_keycallbacl,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            # for i in range(model.nu):
            #     print(f"{i} : {data.qpos[i]:.2f}")

            viewer.sync()

            linear_vels, angular_vels = update_desired_vels_from_keyboard()
            joint_vels = cartesian_velocity_to_joint_velocity(
                model, data, linear_vels, angular_vels, "hand"
            )
            print(joint_vels * model.opt.timestep)
            data.ctrl[:7] += joint_vels * model.opt.timestep


if __name__ == "__main__":
    main()
