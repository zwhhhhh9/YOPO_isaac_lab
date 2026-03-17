"""Dataset loader for YOPO training on collected depth maps."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from yopo_drone.network.models.common import Config, make_config


_GOAL_SAMPLING_MODES = {"gaussian_forward", "uniform_full_yaw", "uniform_box"}


@dataclass(frozen=True)
class _MapSpec:
    map_idx: int
    image_dir: Path
    pose_csv: Path


def _sorted_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        image_dir.glob("img_*.png"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )


def _load_pose_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        positions = []
        quaternions = []
        for row in reader:
            positions.append([float(row["px"]), float(row["py"]), float(row["pz"])])
            quaternions.append([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])])
    return np.asarray(positions, dtype=np.float32), np.asarray(quaternions, dtype=np.float32)


class YopoDataset(Dataset):
    """Training/validation dataset aligned with the original YOPO sampling logic."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        mode: str = "train",
        val_ratio: float = 0.1,
        split_seed: int = 0,
        config: Config | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"train", "valid"}:
            raise ValueError(f"Invalid mode {mode!r}. Expected 'train' or 'valid'.")
        if not (0.0 < float(val_ratio) < 1.0):
            raise ValueError("val_ratio must be in (0, 1).")

        self.cfg = make_config(config, train=True)
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.mode = str(mode)
        self.height = int(self.cfg["image_height"])
        self.width = int(self.cfg["image_width"])

        self.vel_max = float(self.cfg["vel_max_train"])
        self.acc_max = float(self.cfg["acc_max_train"])
        self.vx_lognorm_mean = np.log(1.0 - float(self.cfg["vx_mean_unit"]))
        self.vx_lognorm_sigma = np.log(float(self.cfg["vx_std_unit"]))
        self.v_mean = np.array(
            [self.cfg["vx_mean_unit"], self.cfg["vy_mean_unit"], self.cfg["vz_mean_unit"]],
            dtype=np.float32,
        )
        self.v_std = np.array(
            [self.cfg["vx_std_unit"], self.cfg["vy_std_unit"], self.cfg["vz_std_unit"]],
            dtype=np.float32,
        )
        self.a_mean = np.array(
            [self.cfg["ax_mean_unit"], self.cfg["ay_mean_unit"], self.cfg["az_mean_unit"]],
            dtype=np.float32,
        )
        self.a_std = np.array(
            [self.cfg["ax_std_unit"], self.cfg["ay_std_unit"], self.cfg["az_std_unit"]],
            dtype=np.float32,
        )
        self.goal_length = float(self.cfg["goal_length"])
        self.goal_pitch_std = float(self.cfg["goal_pitch_std"])
        self.goal_yaw_std = float(self.cfg["goal_yaw_std"])
        self.goal_sampling_mode = str(self.cfg["goal_sampling_mode"])
        self.goal_yaw_uniform_min_deg = float(self.cfg["goal_yaw_uniform_min_deg"])
        self.goal_yaw_uniform_max_deg = float(self.cfg["goal_yaw_uniform_max_deg"])
        self.goal_pitch_uniform_min_deg = float(self.cfg["goal_pitch_uniform_min_deg"])
        self.goal_pitch_uniform_max_deg = float(self.cfg["goal_pitch_uniform_max_deg"])
        self._validate_goal_sampling_config()

        self.samples: list[tuple[Path, np.ndarray, np.ndarray, int]] = []
        self._build_samples(val_ratio=float(val_ratio), split_seed=int(split_seed))

    def _discover_maps(self) -> list[_MapSpec]:
        specs: list[_MapSpec] = []

        if (self.dataset_dir / "pose.csv").exists() and (self.dataset_dir / "img").is_dir():
            specs.append(_MapSpec(map_idx=0, image_dir=self.dataset_dir / "img", pose_csv=self.dataset_dir / "pose.csv"))
            return specs

        pose_files = sorted(self.dataset_dir.glob("pose_*.csv"), key=lambda path: int(path.stem.split("_")[-1]))
        for pose_file in pose_files:
            map_idx = int(pose_file.stem.split("_")[-1])
            image_dir = self.dataset_dir / f"img_{map_idx}"
            if image_dir.is_dir():
                specs.append(_MapSpec(map_idx=map_idx, image_dir=image_dir, pose_csv=pose_file))

        if specs:
            return specs

        raise FileNotFoundError(
            f"Failed to discover YOPO dataset structure under {self.dataset_dir}. "
            "Expected pose.csv + img/ or pose_<n>.csv + img_<n>/."
        )

    def _build_samples(self, *, val_ratio: float, split_seed: int) -> None:
        map_specs = self._discover_maps()
        print(f"Loading {self.mode} dataset from {self.dataset_dir}")
        for spec in map_specs:
            image_paths = _sorted_image_paths(spec.image_dir)
            positions, quaternions = _load_pose_csv(spec.pose_csv)
            if len(image_paths) != len(positions):
                raise ValueError(
                    f"Image/pose count mismatch in map {spec.map_idx}: "
                    f"{len(image_paths)} images vs {len(positions)} poses."
                )

            rng = np.random.RandomState(split_seed)
            indices = np.arange(len(image_paths))
            rng.shuffle(indices)
            if len(indices) <= 1:
                train_indices = indices
                val_indices = indices
            else:
                val_count = max(1, int(math.ceil(len(indices) * val_ratio)))
                val_count = min(val_count, len(indices) - 1)
                val_indices = indices[:val_count]
                train_indices = indices[val_count:]
            selected_indices = train_indices if self.mode == "train" else val_indices

            for idx in selected_indices:
                self.samples.append((image_paths[idx], positions[idx], quaternions[idx], spec.map_idx))

        print(
            f"{self.mode.capitalize()} dataset ready: "
            f"samples={len(self.samples)}, image_size=({self.height}, {self.width})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, position, quaternion_wxyz, map_idx = self.samples[index]

        image = np.array(Image.open(image_path), dtype=np.float32)
        if image.shape != (self.height, self.width):
            image = np.array(
                Image.fromarray(image.astype(np.uint16), mode="I;16").resize((self.width, self.height), resample=Image.NEAREST),
                dtype=np.float32,
            )
        image = np.expand_dims(image / 65535.0, axis=0).astype(np.float32)

        rotation_wb = R.from_quat(
            [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]
        )
        yaw_pitch_roll = rotation_wb.as_euler("ZYX", degrees=False)
        rotation_bw_level = R.from_euler("ZYX", [0.0, yaw_pitch_roll[1], yaw_pitch_roll[2]], degrees=False).inv()

        vel_w, acc_w = self._get_random_state()
        vel_b = rotation_bw_level.apply(vel_w)
        acc_b = rotation_bw_level.apply(acc_w)
        goal_w = self._get_random_goal()
        goal_b = rotation_bw_level.apply(goal_w)

        obs_body = np.hstack((vel_b, acc_b, goal_b)).astype(np.float32)
        rot_wb = rotation_wb.as_matrix().astype(np.float32)

        return (
            torch.from_numpy(image),
            torch.from_numpy(position.astype(np.float32)),
            torch.from_numpy(rot_wb),
            torch.from_numpy(obs_body),
            torch.tensor(map_idx, dtype=torch.long),
        )

    def _get_random_state(self) -> tuple[np.ndarray, np.ndarray]:
        while True:
            vel = self.vel_max * (self.v_mean + self.v_std * np.random.standard_normal(3))
            right_skewed_vx = -1.0
            while right_skewed_vx < 0.0:
                right_skewed_vx = self.vel_max * np.random.lognormal(mean=self.vx_lognorm_mean, sigma=self.vx_lognorm_sigma)
                right_skewed_vx = -right_skewed_vx + 1.2 * self.vel_max
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.vel_max:
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.standard_normal(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:
                break
        return vel.astype(np.float32), acc.astype(np.float32)

    def _get_random_goal(self) -> np.ndarray:
        goal_pitch_angle = self._sample_goal_pitch_angle_rad()
        goal_yaw_angle = self._sample_goal_yaw_angle_rad()
        goal_w_dir = np.array(
            [
                np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
                np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle),
                np.sin(goal_pitch_angle),
            ],
            dtype=np.float32,
        )
        random_near = np.random.random()
        if random_near < 0.1:
            goal_w_dir = goal_w_dir * float(random_near * 10.0)
        return (self.goal_length * goal_w_dir).astype(np.float32)

    def _sample_goal_pitch_angle_rad(self) -> float:
        if self.goal_sampling_mode == "uniform_box":
            pitch_deg = np.random.uniform(self.goal_pitch_uniform_min_deg, self.goal_pitch_uniform_max_deg)
        else:
            pitch_deg = np.random.normal(0.0, self.goal_pitch_std)
        return float(np.radians(pitch_deg))

    def _sample_goal_yaw_angle_rad(self) -> float:
        if self.goal_sampling_mode in {"uniform_full_yaw", "uniform_box"}:
            yaw_deg = np.random.uniform(self.goal_yaw_uniform_min_deg, self.goal_yaw_uniform_max_deg)
        else:
            yaw_deg = np.random.normal(0.0, self.goal_yaw_std)
        return float(np.radians(yaw_deg))

    def _validate_goal_sampling_config(self) -> None:
        if self.goal_sampling_mode not in _GOAL_SAMPLING_MODES:
            valid_modes = ", ".join(sorted(_GOAL_SAMPLING_MODES))
            raise ValueError(
                f"Unsupported goal_sampling_mode={self.goal_sampling_mode!r}. "
                f"Expected one of: {valid_modes}."
            )
        if self.goal_yaw_uniform_min_deg > self.goal_yaw_uniform_max_deg:
            raise ValueError("goal_yaw_uniform_min_deg must be <= goal_yaw_uniform_max_deg.")
        if self.goal_pitch_uniform_min_deg > self.goal_pitch_uniform_max_deg:
            raise ValueError("goal_pitch_uniform_min_deg must be <= goal_pitch_uniform_max_deg.")
