"""Runtime helpers for closed-loop YOPO policy evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as f
from scipy.spatial.transform import Rotation as R

try:
    import cv2
except ImportError:
    cv2 = None

from yopo_drone.network.models.common import Config, make_config
from yopo_drone.network.models.test.yopo_test_model import YopoTestModel


@dataclass(frozen=True)
class YopoPolicyPlan:
    mission_goal_world: np.ndarray
    segment_goal_world: np.ndarray
    segment_velocity_world: np.ndarray
    segment_acceleration_world: np.ndarray
    score: float
    action_id: int
    segment_duration: float
    candidate_segments: tuple["YopoPolicyCandidate", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class YopoPolicyCandidate:
    segment_goal_world: np.ndarray
    segment_velocity_world: np.ndarray
    segment_acceleration_world: np.ndarray
    score: float
    action_id: int
    segment_duration: float


def _summary_path_candidates(checkpoint_path: Path) -> Iterable[Path]:
    stem = checkpoint_path.stem
    yield checkpoint_path.with_name(f"{stem}_summary.json")
    if stem.endswith("_best"):
        base_stem = stem[: -len("_best")]
        yield checkpoint_path.with_name(f"{base_stem}_summary.json")


def _load_checkpoint_summary(checkpoint_path: Path) -> dict[str, object] | None:
    for summary_path in _summary_path_candidates(checkpoint_path):
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _load_checkpoint_config(checkpoint_path: Path) -> tuple[dict[str, object], str]:
    summary = _load_checkpoint_summary(checkpoint_path)
    if isinstance(summary, dict):
        config_payload = summary.get("config")
        if isinstance(config_payload, dict):
            return dict(config_payload), "checkpoint_summary"
    return {}, "builtin_default"


def _resolve_dataset_dir(checkpoint_path: Path, dataset_dir: str | Path | None) -> Path:
    if dataset_dir is not None and str(dataset_dir).strip():
        return Path(dataset_dir).expanduser().resolve()
    summary = _load_checkpoint_summary(checkpoint_path)
    if isinstance(summary, dict):
        resolved = summary.get("dataset_dir")
        if resolved:
            return Path(resolved).expanduser().resolve()
    raise FileNotFoundError(
        f"Unable to resolve dataset_dir for checkpoint {checkpoint_path}. "
        "Pass --yopo_policy_dataset_dir explicitly or keep the *_summary.json file next to the checkpoint."
    )


def _load_dataset_metadata(dataset_dir: Path) -> dict[str, object] | None:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_dataset_camera_pitch_deg(metadata: dict[str, object] | None) -> float | None:
    if not isinstance(metadata, dict):
        return None
    candidates: list[object] = []
    if "camera_pitch_deg" in metadata:
        candidates.append(metadata["camera_pitch_deg"])
    args_payload = metadata.get("args")
    if isinstance(args_payload, dict) and "camera_pitch_deg" in args_payload:
        candidates.append(args_payload["camera_pitch_deg"])
    for value in candidates:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_camera_pitch_deg(dataset_dir: Path, camera_pitch_deg: float | None) -> tuple[float, float | None, str]:
    dataset_camera_pitch_deg = _extract_dataset_camera_pitch_deg(_load_dataset_metadata(dataset_dir))
    if camera_pitch_deg is not None:
        return float(camera_pitch_deg), dataset_camera_pitch_deg, "cli"
    if dataset_camera_pitch_deg is not None:
        return dataset_camera_pitch_deg, dataset_camera_pitch_deg, "dataset_metadata"
    return 0.0, None, "default_0.0"


def _load_pose_positions(path: Path) -> np.ndarray:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        positions = [
            [float(row["px"]), float(row["py"]), float(row["pz"])]
            for row in reader
        ]
    if not positions:
        raise ValueError(f"No pose rows found in {path}")
    return np.asarray(positions, dtype=np.float32)


def _discover_goal_positions(dataset_dir: Path) -> np.ndarray:
    pose_paths: list[Path] = []
    single_pose = dataset_dir / "pose.csv"
    if single_pose.exists():
        pose_paths.append(single_pose)
    else:
        pose_paths.extend(sorted(dataset_dir.glob("pose_*.csv"), key=lambda path: int(path.stem.split("_")[-1])))
    if not pose_paths:
        raise FileNotFoundError(f"No pose.csv or pose_<n>.csv found under {dataset_dir}")
    return np.concatenate([_load_pose_positions(path) for path in pose_paths], axis=0)


class YopoPolicyRuntime:
    """Loads a trained YOPO checkpoint and produces segment plans for receding-horizon control."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        dataset_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        compact_backbone: bool = False,
        max_depth_dist: float = 20.0,
        camera_pitch_deg: float | None = None,
        velocity: float | None = None,
        goal_seed: int = 0,
        goal_xy_bounds: tuple[float, float, float, float] | None = None,
        goal_trunk_obstacles: np.ndarray | None = None,
        goal_trunk_clearance: float = 0.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"YOPO checkpoint not found: {self.checkpoint_path}")

        self.dataset_dir = _resolve_dataset_dir(self.checkpoint_path, dataset_dir)
        self.goal_positions = _discover_goal_positions(self.dataset_dir)
        self.goal_rng = np.random.default_rng(int(goal_seed))
        self.min_depth_dist = 0.04
        self.max_depth_dist = max(float(max_depth_dist), 1e-6)
        self.goal_xy_bounds = None if goal_xy_bounds is None else tuple(float(value) for value in goal_xy_bounds)
        self.goal_trunk_clearance = max(float(goal_trunk_clearance), 0.0)
        if goal_trunk_obstacles is None:
            self.goal_trunk_obstacles = np.zeros((0, 3), dtype=np.float32)
        else:
            self.goal_trunk_obstacles = np.asarray(goal_trunk_obstacles, dtype=np.float32).reshape(-1, 3)
        self._goal_clearance_warning_emitted = False

        resolved_device = torch.device(device) if device not in (None, "auto") else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = resolved_device

        config_payload, config_source = _load_checkpoint_config(self.checkpoint_path)
        config_payload["dataset_path"] = str(self.dataset_dir)
        if velocity is not None:
            config_payload["velocity"] = float(velocity)
        self.config_source = config_source if velocity is None else f"{config_source}+velocity_override"
        self.config: Config = make_config(config_payload, train=False)
        self.image_height = int(self.config["image_height"])
        self.image_width = int(self.config["image_width"])
        resolved_camera_pitch_deg, dataset_camera_pitch_deg, camera_pitch_source = _resolve_camera_pitch_deg(
            self.dataset_dir,
            camera_pitch_deg,
        )
        self.camera_pitch_deg = float(resolved_camera_pitch_deg)
        self.dataset_camera_pitch_deg = dataset_camera_pitch_deg
        self.camera_pitch_source = camera_pitch_source
        self.camera_pitch_matches_dataset = (
            dataset_camera_pitch_deg is None or abs(self.camera_pitch_deg - dataset_camera_pitch_deg) <= 1e-6
        )
        self.rotation_bc = R.from_euler("ZYX", [0.0, self.camera_pitch_deg, 0.0], degrees=True).as_matrix()

        self.model = YopoTestModel(
            compact_backbone=compact_backbone,
            config=self.config,
            device=self.device,
        )
        self.model.load_checkpoint(self.checkpoint_path, map_location=self.device)
        self.model.eval()
        self.segment_duration = float(self.model.lattice_primitive.segment_time)
        self.max_speed = float(self.model.lattice_primitive.vel_max)
        self.max_acceleration = float(self.model.lattice_primitive.acc_max)

    def sample_goal(self, *, start_pos: np.ndarray, min_distance: float = 5.0) -> np.ndarray:
        start = np.asarray(start_pos, dtype=np.float32).reshape(3)
        candidate_positions = self._goal_candidate_positions()
        if self.goal_xy_bounds is not None and len(candidate_positions) > 0:
            candidate_positions = candidate_positions.copy()
            candidate_positions[:, 2] = float(start[2])
        candidate_positions = self._filter_goal_candidates_by_trunk_clearance(candidate_positions)
        candidate_indices = np.arange(len(candidate_positions))
        self.goal_rng.shuffle(candidate_indices)
        min_distance = max(float(min_distance), 0.0)
        for index in candidate_indices:
            candidate = candidate_positions[index]
            if np.linalg.norm(candidate - start) >= min_distance:
                return candidate.astype(np.float32).copy()
        if len(candidate_positions) == 0:
            return self._sample_bounded_goal_fallback(start=start, min_distance=min_distance)
        distances = np.linalg.norm(candidate_positions - start[None, :], axis=1)
        farthest_index = int(np.argmax(distances))
        return candidate_positions[farthest_index].astype(np.float32).copy()

    def _sample_planar_goal(self, *, start: np.ndarray, min_distance: float) -> np.ndarray:
        if self.goal_xy_bounds is None:
            raise RuntimeError("Planar goal sampling requested without goal_xy_bounds.")
        min_x, max_x, min_y, max_y = self.goal_xy_bounds
        goal_z = float(start[2])
        for _ in range(128):
            candidate = np.array(
                [
                    self.goal_rng.uniform(min_x, max_x),
                    self.goal_rng.uniform(min_y, max_y),
                    goal_z,
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(candidate - start) >= min_distance and self._candidate_has_trunk_clearance(candidate):
                return candidate

        corners = np.array(
            [
                [min_x, min_y, goal_z],
                [min_x, max_y, goal_z],
                [max_x, min_y, goal_z],
                [max_x, max_y, goal_z],
            ],
            dtype=np.float32,
        )
        return self._select_fallback_goal(corners, start=start)

    def _goal_candidate_positions(self) -> np.ndarray:
        if self.goal_xy_bounds is None:
            return self.goal_positions
        min_x, max_x, min_y, max_y = self.goal_xy_bounds
        mask = (
            (self.goal_positions[:, 0] >= min_x)
            & (self.goal_positions[:, 0] <= max_x)
            & (self.goal_positions[:, 1] >= min_y)
            & (self.goal_positions[:, 1] <= max_y)
        )
        return self.goal_positions[mask]

    def _sample_bounded_goal_fallback(self, *, start: np.ndarray, min_distance: float) -> np.ndarray:
        if self.goal_xy_bounds is None:
            raise RuntimeError("Bounded goal fallback requested without goal_xy_bounds.")
        min_x, max_x, min_y, max_y = self.goal_xy_bounds
        z_value = float(start[2])
        for _ in range(128):
            candidate = np.array(
                [
                    self.goal_rng.uniform(min_x, max_x),
                    self.goal_rng.uniform(min_y, max_y),
                    z_value,
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(candidate - start) >= min_distance and self._candidate_has_trunk_clearance(candidate):
                return candidate

        corners = np.array(
            [
                [min_x, min_y, z_value],
                [min_x, max_y, z_value],
                [max_x, min_y, z_value],
                [max_x, max_y, z_value],
            ],
            dtype=np.float32,
        )
        return self._select_fallback_goal(corners, start=start)

    def _filter_goal_candidates_by_trunk_clearance(self, candidates: np.ndarray) -> np.ndarray:
        if len(candidates) == 0 or self.goal_trunk_obstacles.size == 0:
            return candidates
        mask = np.array([self._candidate_has_trunk_clearance(candidate) for candidate in candidates], dtype=bool)
        return candidates[mask]

    def _candidate_has_trunk_clearance(self, candidate: np.ndarray) -> bool:
        if self.goal_trunk_obstacles.size == 0:
            return True
        delta_xy = self.goal_trunk_obstacles[:, :2] - candidate[None, :2]
        distances = np.linalg.norm(delta_xy, axis=1)
        min_allowed = self.goal_trunk_obstacles[:, 2] + self.goal_trunk_clearance
        return bool(np.all(distances >= min_allowed))

    def _candidate_trunk_clearance(self, candidate: np.ndarray) -> float:
        if self.goal_trunk_obstacles.size == 0:
            return float("inf")
        delta_xy = self.goal_trunk_obstacles[:, :2] - candidate[None, :2]
        distances = np.linalg.norm(delta_xy, axis=1)
        min_allowed = self.goal_trunk_obstacles[:, 2] + self.goal_trunk_clearance
        return float(np.min(distances - min_allowed))

    def _select_fallback_goal(self, candidates: np.ndarray, *, start: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 3)
        if len(candidates) == 0:
            raise RuntimeError("No goal candidates are available for fallback sampling.")
        safe_candidates = self._filter_goal_candidates_by_trunk_clearance(candidates)
        source = safe_candidates if len(safe_candidates) > 0 else candidates
        distances = np.linalg.norm(source - start[None, :], axis=1)
        if len(safe_candidates) == 0 and not self._goal_clearance_warning_emitted:
            best_clearance = max(self._candidate_trunk_clearance(candidate) for candidate in candidates)
            print(
                "[Warn] YOPO goal sampling could not satisfy the configured trunk-clearance constraint. "
                f"Falling back to the farthest bounded candidate with best available clearance={best_clearance:.3f} m.",
                flush=True,
            )
            self._goal_clearance_warning_emitted = True
        return source[int(np.argmax(distances))].astype(np.float32).copy()

    def infer_segment(
        self,
        *,
        depth_image_m: np.ndarray,
        start_pos_world: np.ndarray,
        start_quat_wxyz: np.ndarray,
        start_vel_world: np.ndarray,
        start_acc_world: np.ndarray,
        mission_goal_world: np.ndarray,
        goal_reference_pos_world: np.ndarray | None = None,
        candidate_limit: int | None = None,
    ) -> YopoPolicyPlan:
        start_pos_world = np.asarray(start_pos_world, dtype=np.float64).reshape(3)
        start_quat_wxyz = np.asarray(start_quat_wxyz, dtype=np.float64).reshape(4)
        start_vel_world = np.asarray(start_vel_world, dtype=np.float64).reshape(3)
        start_acc_world = np.asarray(start_acc_world, dtype=np.float64).reshape(3)
        mission_goal_world = np.asarray(mission_goal_world, dtype=np.float64).reshape(3)
        if goal_reference_pos_world is None:
            goal_reference_pos_world = start_pos_world
        else:
            goal_reference_pos_world = np.asarray(goal_reference_pos_world, dtype=np.float64).reshape(3)

        rotation_wb = R.from_quat(
            [start_quat_wxyz[1], start_quat_wxyz[2], start_quat_wxyz[3], start_quat_wxyz[0]]
        ).as_matrix()
        rotation_wc = rotation_wb @ self.rotation_bc
        rotation_cw = rotation_wc.T

        vel_c = rotation_cw @ start_vel_world
        acc_c = rotation_cw @ start_acc_world
        goal_c = rotation_cw @ (mission_goal_world - goal_reference_pos_world)
        obs_body = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)[None, :]

        depth_tensor = self._prepare_depth_tensor(depth_image_m)
        obs_tensor = torch.from_numpy(obs_body).to(device=self.device, dtype=torch.float32)
        endstate_batch, score_batch, action_ids = self.model.predict(
            depth_tensor,
            obs_tensor,
            return_all_predictions=True,
        )

        score_flat = np.asarray(score_batch[0], dtype=np.float32).reshape(-1)
        action_id_flat = np.asarray(action_ids[0], dtype=np.int64).reshape(-1)
        endstate_c = np.asarray(endstate_batch[0], dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
        endstate_w = np.einsum("ij,njk->nik", rotation_wc, endstate_c)
        segment_goal_world = start_pos_world[None, :] + endstate_w[:, :, 0]
        segment_velocity_world = endstate_w[:, :, 1]
        segment_acceleration_world = endstate_w[:, :, 2]

        score_order = np.argsort(score_flat, kind="stable")
        best_index = int(score_order[0])

        if candidate_limit is None:
            candidate_order = score_order
        else:
            candidate_order = score_order[: max(int(candidate_limit), 0)]

        candidate_segments = tuple(
            YopoPolicyCandidate(
                segment_goal_world=segment_goal_world[index].astype(np.float32),
                segment_velocity_world=segment_velocity_world[index].astype(np.float32),
                segment_acceleration_world=segment_acceleration_world[index].astype(np.float32),
                score=float(score_flat[index]),
                action_id=int(action_id_flat[index]),
                segment_duration=self.segment_duration,
            )
            for index in candidate_order
        )

        return YopoPolicyPlan(
            mission_goal_world=mission_goal_world.astype(np.float32),
            segment_goal_world=segment_goal_world[best_index].astype(np.float32),
            segment_velocity_world=segment_velocity_world[best_index].astype(np.float32),
            segment_acceleration_world=segment_acceleration_world[best_index].astype(np.float32),
            score=float(score_flat[best_index]),
            action_id=int(action_id_flat[best_index]),
            segment_duration=self.segment_duration,
            candidate_segments=candidate_segments,
        )

    def _prepare_depth_tensor(self, depth_image_m: np.ndarray) -> torch.Tensor:
        depth = np.asarray(depth_image_m, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth image, got shape {tuple(depth.shape)}")

        depth = np.clip(
            np.nan_to_num(depth, nan=self.max_depth_dist, posinf=self.max_depth_dist, neginf=0.0),
            0.0,
            self.max_depth_dist,
        )
        if depth.shape != (self.image_height, self.image_width):
            depth = np.asarray(
                f.interpolate(
                    torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
                    size=(self.image_height, self.image_width),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy(),
                dtype=np.float32,
            )

        depth = depth / self.max_depth_dist
        invalid_mask = np.isnan(depth) | (depth < self.min_depth_dist / self.max_depth_dist)
        if np.any(invalid_mask):
            if cv2 is not None:
                inpaint_input = np.uint8(np.clip(depth, 0.0, 1.0) * 255.0)
                depth = cv2.inpaint(inpaint_input, np.uint8(invalid_mask), 1, cv2.INPAINT_NS).astype(np.float32) / 255.0
            else:
                depth = np.where(invalid_mask, 0.0, depth)

        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        return depth_tensor.to(device=self.device, dtype=torch.float32)
