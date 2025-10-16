import mujoco
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
import mujoco.viewer
import time

import pick_and_place_env

try:
    # Load dataset
    dataset = LeRobotDataset(
        repo_id="LuminousLlama/mojoco_pick_and_place",
    )
    print("done loading")

    # Load MuJoCo
    model = pick_and_place_env.create_env()
    data = mujoco.MjData(model)

    # Replay episode 0
    episode_idx = 0
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()

    first_frame = dataset[int(from_idx)]
    # last_frame = dataset[int(to_idx - 1)]
    data.qpos[:9] = first_frame["observation.state"].numpy()
    data.qpos[9:] = first_frame["observation.objs"].numpy()
    data.ctrl[:] = first_frame["action"].numpy()

    mujoco.mj_forward(model, data)
    # print(first_frame["observation.cube"])
    # print(last_frame["observation.cube"])

    # print(dataset[0])
    # print(dataset[to_idx - 1])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for step_idx in range(int(from_idx), int(to_idx)):
                frame = dataset[step_idx]

                # Force state to match recording (overrides physics)
                data.qpos[:9] = frame["observation.state"].numpy()
                data.qpos[9:] = frame["observation.objs"].numpy()

                # Apply recorded action
                data.ctrl[:] = frame["action"].numpy()

                # Recompute derived quantities
                mujoco.mj_forward(model, data)

                viewer.sync()
                # time.sleep(0.2)
                print(
                    f"Frame {step_idx}:"  # state error = {np.linalg.norm(current_state - recorded_state):.4f}"
                )
except Exception as e:
    print(e)
