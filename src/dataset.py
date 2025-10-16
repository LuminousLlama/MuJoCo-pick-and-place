from lerobot.datasets.lerobot_dataset import LeRobotDataset


def create_new_dataset() -> LeRobotDataset:
    import shutil
    from pathlib import Path

    repo_id = "LuminousLlama/mojoco_pick_and_place"

    dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_id

    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=500,
        robot_type="panda",
        features={
            "observation.state": {
                "dtype": "float64",
                "shape": (9,),
            },
            "observation.cube": {
                "dtype": "float64",
                "shape": (7,),
            },
            "action": {
                "dtype": "float64",
                "shape": (8,),
            },
        },
    )

    return dataset
