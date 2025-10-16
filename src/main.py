import mujoco
import mujoco.viewer
import pick_and_place_env
from pynput import keyboard
from sys import exit
import numpy as np
import mujoco_viewer

import time
import dataset

import torch

import lerobot.datasets


active_keys = set()

LINEAR_SPEED = 0.15
ANGULAR_SPEED = 0.3
DAMPING = 0.01


def on_press(key):
    try:
        active_keys.add(key.char)
    except:
        return


def on_release(key):
    if key == keyboard.Key.shift:
        active_keys.clear()
    else:
        try:
            ch = key.char
            ch = ch.upper()
            active_keys.discard(key.char)
        except:
            return


def update_desired_vels_from_keyboard():
    linear_vels = [0.0, 0.0, 0.0]
    angular_vels = [0.0, 0.0, 0.0]
    gripper_status = 0

    if "W" in active_keys:
        linear_vels[0] += LINEAR_SPEED
    if "S" in active_keys:
        linear_vels[0] -= LINEAR_SPEED
    if "A" in active_keys:
        linear_vels[1] += LINEAR_SPEED
    if "D" in active_keys:
        linear_vels[1] -= LINEAR_SPEED
    if "Q" in active_keys:
        linear_vels[2] += LINEAR_SPEED
    if "E" in active_keys:
        linear_vels[2] -= LINEAR_SPEED

    # Rotation (Angular velocity)
    # pitch is differently flipped from others so that "i" makes it go up not down)
    if "I" in active_keys:
        angular_vels[1] -= ANGULAR_SPEED  # Pitch up (rotate around Y-)
    if "K" in active_keys:
        angular_vels[1] += ANGULAR_SPEED  # Pitch down (rotate around Y+)
    if "J" in active_keys:
        angular_vels[2] += ANGULAR_SPEED  # Yaw left (rotate around Z+)
    if "L" in active_keys:
        angular_vels[2] -= ANGULAR_SPEED  # Yaw right (rotate around Z-)
    if "U" in active_keys:
        angular_vels[0] += ANGULAR_SPEED  # Roll left (rotate around X+)
    if "O" in active_keys:
        angular_vels[0] -= ANGULAR_SPEED  # Roll right (rotate around X-)

    # pos = open, neg = close
    if "Z" in active_keys:
        gripper_status += -1
    if "X" in active_keys:
        gripper_status += 1

    return linear_vels, angular_vels, gripper_status


def cartesian_velocity_to_joint_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    linear_velocity,
    angular_velocity,
    end_effector_body_name,
    damping=0.01,
):
    if all(val == 0 for val in linear_velocity) and all(
        val == 0 for val in angular_velocity
    ):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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


def main():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    dataset_lerobot = dataset.create_new_dataset()

    model = pick_and_place_env.create_env()
    data = mujoco.MjData(model)

    # elliptic better for preventing slip
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # very important to use this integrator for ridge body manipulation
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # enable if observe tunneling / penetration issues
    # model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD

    model.key_qpos[0] = np.array(
        [
            -0.0261624,
            0.167415,
            0.0135335,
            -2.40325,
            0.0615261,
            2.57368,
            0.754441,
            0.0399996,
            0.0399993,
            -0.3,
            0.0,
            0.844915,
            1,
            0.0,
            0.0,
            0.0,
        ]
    )
    model.key_ctrl[0] = np.array(
        [-0.0341773, 0.160622, 0.0365913, -2.39816, 0.0611585, 2.57367, 0.754311, 255]
    )

    # hand tuned. Gainprm depends on the control signal not the length
    # ctrl signal 0 - 255
    model.actuator_gainprm[7, 0] = 0.17
    model.actuator_biasprm[7, 0] = 0
    # this is very important. more negative = more clamping force
    model.actuator_biasprm[7, 1] = -800
    model.actuator_biasprm[7, 2] = -250

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        try:
            while viewer.is_running():
                if "R" in active_keys:
                    mujoco.mj_resetDataKeyframe(model, data, 0)

                # current qpos states
                robot_states = data.qpos[0:9].copy()
                cube_states = data.qpos[9:16].copy()

                # teleop input
                linear_vels, angular_vels, gripper_status = (
                    update_desired_vels_from_keyboard()
                )
                joint_vels = cartesian_velocity_to_joint_velocity(
                    model, data, linear_vels, angular_vels, "hand"
                )

                # control signals
                joint_ctrls = data.ctrl[:7].copy() + joint_vels * model.opt.timestep
                gripper_ctrl = None
                if gripper_status == 0:
                    gripper_ctrl = data.ctrl[7]
                elif gripper_status == -1:
                    gripper_ctrl = 0
                elif gripper_status == 1:
                    gripper_ctrl = 255
                else:
                    print("unexpected gripper status")
                    viewer.close()
                    exit(1)

                action = np.append(joint_ctrls, gripper_ctrl).copy()

                dataset_lerobot.add_frame(
                    task="pick_and_place",
                    frame={
                        "observation.state": torch.from_numpy(robot_states),
                        "observation.cube": torch.from_numpy(cube_states),
                        "action": torch.from_numpy(action),
                    },
                    timestamp=data.time,
                )

                data.ctrl[:] = action
                mujoco.mj_step(model, data)
                viewer.sync()

                # print(
                #     f"Force: {data.actuator_force[7]:.8f} Length: {data.actuator_length[7]:.4f} vel: {data.actuator_velocity[7]:.4f}"
                # )
        except KeyboardInterrupt:
            dataset_lerobot.save_episode()
            viewer.close()


if __name__ == "__main__":
    main()
