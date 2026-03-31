"""
Minimal example: convert dataset to the LeRobot format.

CLI Example (using the *arrange_flowers* task as an example):
    python convert_libero_to_lerobot.py \
        --repo-name arrange_flowers_repo \
        --raw-dataset /path/to/arrange_flowers \
        --frame-interval 1 \

Notes:
- If you plan to push to the Hugging Face Hub later, handle that outside this script.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
# from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
from scipy.spatial.transform import Rotation as R



def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def create_lerobot_dataset(
    repo_name: str,
    robot_type: str,
    fps: float,
    height: int,
    width: int,
) -> LeRobotDataset:
    """
    Create a LeRobot dataset with custom feature schema
    """
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features={
            "global_image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            },
            "right_image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,), # for ee_pose and gripper width
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,), # for ee_pose and gripper width
                "names": ["actions"],
            },
        },
        image_writer_threads=32,
        image_writer_processes=16,
    )
    return dataset

def image_bytes_to_np(img_bytes: bytes) -> np.ndarray:
    img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def tcp_pose_7d_to_6d(tcp_pose_7d):
    # tcp_pose_7d: [x, y, z, qx, qy, qz, qw]
    tcp_pose_7d = np.asarray(tcp_pose_7d, dtype=np.float64)
    xyz = tcp_pose_7d[:3]
    quat_xyzw = tcp_pose_7d[3:7]   # 注意顺序必须是 x,y,z,w

    rpy = R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)  # 弧度
    return np.concatenate([xyz, rpy], axis=0)  # [x,y,z,roll,pitch,yaw]


def process_episode_dir(
    episode_path: Path,
    dataset: LeRobotDataset,
) -> None:
    """
    Process a single episode directory and append frames to the given dataset.

    episode_path : Path
        Episode directory containing `states/states.jsonl` and `videos/*.mp4`.
    dataset : LeRobotDataset
        Target dataset to which frames are added.
    frame_interval : int
        Sampling stride (>=1).
    prompt : str
        Language instruction of this episode.
    """

    """
    数据schema
    - task_description: str, 任务prompt
    - observation/images/left_cam: bytes
    - observation/images/right_cam: bytes
    - observation/images/wrist_cam: bytes
    - observation/images/left_cam_shape: str
    - observation/images/right_cam_shape: str
    - observation/images/wrist_cam_shape: str

    - observation/state/tcp_pose: 末端位姿，7维：[x, y, z, qx, qy, qz, qw]，前3维是位置（米），后4维是四元数姿态
    - observation/state/tcp_vel: 末端速度，6维：[vx, vy, vz, wx, wy, wz]，线速度单位 m/s，角速度单位 rad/s
    - observation/state/tcp_force: 末端受力，3维：[fx, fy, fz]，单位 Newton（N）
    - observation/state/tcp_torque: 末端力矩，3维：[tx, ty, tz]，单位 N·m
    - observation/state/gripper_pose: 夹爪开合，1维（或标量），约定是 0=闭合, 1=张开（中间值表示部分开合）
    """

    import pandas as pd
    df = pd.read_parquet(episode_path)
    # TODO: frame_interval = 1
    last_row = None
    for idx, row in df.iterrows():
        if last_row is None:
            last_row = row
            continue

        global_image = image_bytes_to_np(row["observation/images/left_cam"])
        right_image = image_bytes_to_np(row["observation/images/right_cam"])
        wrist_image = image_bytes_to_np(row["observation/images/wrist_cam"])

        last_tcp_pose = last_row["observation/state/tcp_pose"]
        last_gripper_pose = last_row["observation/state/gripper_pose"]
        last_pose = np.concatenate([tcp_pose_7d_to_6d(last_tcp_pose), [last_gripper_pose]])

        tcp_pose = row["observation/state/tcp_pose"]
        gripper_pose = row["observation/state/gripper_pose"]
        pose = np.concatenate([tcp_pose_7d_to_6d(tcp_pose), [gripper_pose]])

        dataset.add_frame(
            {
                "global_image": global_image,
                "right_image": right_image,
                "wrist_image": wrist_image,
                "state": last_pose.astype(np.float32, copy=False),
                "actions": pose.astype(np.float32, copy=False),
            },
            task=row["task_description"],
        )
        last_row = row

    # dataset.save_episode(task=prompt)
    dataset.save_episode()


def main(
    repo_name: str,
    raw_dataset: Path,
    output_dir: Path,
    overwrite_repo: bool = False,
) -> None:
    """
    Convert a dataset directory into LeRobot format.

    repo_name : str
        Output repo/dataset name (saved under $LEROBOT_HOME / repo_name).
    raw_dataset : Path
        Path to the raw dataset root directory.
    overwrite_repo : bool, default=False
        If True, remove the existing dataset directory before writing.
    """
    # overwrite repo
    dst_dir = output_dir
    if overwrite_repo and dst_dir.exists():
        print(f"removing existing dataset at {dst_dir}")
        shutil.rmtree(dst_dir)

    robot_type = "franka"
    video_info = {}
    video_info["width"]  = 420  # TODO: derive from task_info or actual videos
    video_info["height"] = 240
    fps = 30

    # Create dataset, define feature in the form you need.
    # - proprio is stored in `state` and actions in `action`
    # - LeRobot assumes that dtype of image data is `image`
    dataset = create_lerobot_dataset(
        repo_name=repo_name,
        robot_type=robot_type,
        fps=fps,
        height=video_info["height"],
        width=video_info["width"],
    )

    # populate the dataset to lerobot dataset
    for episode_path in raw_dataset.rglob("*"):
        print(f"Processing episode: {episode_path.name}")
        process_episode_dir(
            episode_path=episode_path,
            dataset=dataset,
        )

    # dataset.consolidate(run_compute_stats=False)
    if hasattr(dataset, "consolidate"):
        dataset.consolidate(run_compute_stats=False)
    print(f"Done. Dataset saved to: {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert private dataset to LeRobot format."
    )
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Name of the output dataset (under $LEROBOT_HOME).",
    )
    parser.add_argument(
        "--raw-dataset",
        required=True,
        type=str,
        help="Path to the raw dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--overwrite-repo",
        action="store_true",
        help="Remove existing output directory if it exists.",
    )
    args = parser.parse_args()

    main(
        repo_name=args.repo_name,
        raw_dataset=Path(args.raw_dataset),
        output_dir=Path(args.output_dir),
        overwrite_repo=args.overwrite_repo,
    )
