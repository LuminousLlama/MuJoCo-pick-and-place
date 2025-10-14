import mujoco
import mujoco.viewer
import pick_and_place_env


model = pick_and_place_env.create_env()
data = mujoco.MjData(model)


def main():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            for i in range(model.nu):
                print(f"{i} : {data.qpos[i]:.2f}")

            viewer.sync()


if __name__ == "__main__":
    main()
