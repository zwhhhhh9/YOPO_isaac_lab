#!/usr/bin/env python3
"""Collect YOPO-style depth datasets inside Isaac Lab.

The output layout intentionally mirrors the original YOPO dataset generator:

    yopo_drone/network/data_train/<timestamp>/
      img/
        img_0.png
        img_1.png
        ...
      pose.csv
      pointcloud.ply
      metadata.json

Each dataset run gets its own timestamped directory so the collected assets stay
grouped together. The exported point cloud includes sampled obstacle surfaces and,
by default, a ground plane so YOPO's safety loss can treat the floor as an
obstacle just like the original pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

try:
    import cv2
except ImportError:  # pragma: no cover - depends on runtime image
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on runtime image
    Image = None

from yopo_drone.env.drone_env_editor import (
    _derive_depth_tiled_camera_prim_path,
    _ensure_isaaclab_pythonpath,
    _ensure_pxr_imported,
    _project_root,
    initialize_scene_from_editor,
)
from yopo_drone.env.random_forest_scene import (
    RandomForestSceneCfg,
    add_random_forest_arguments,
    add_random_forest_scene,
    build_random_forest_cfg_from_args,
)
from yopo_drone.env.random_forest_scene import _generate_tile_positions, _tile_seed

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - depends on runtime image
    cKDTree = None

try:
    from isaaclab.app import AppLauncher
except ImportError:
    _ensure_isaaclab_pythonpath()
    from isaaclab.app import AppLauncher


_DEFAULT_DATA_ROOT = _project_root() / "yopo_drone" / "network" / "data_train"


@dataclass(slots=True)
class TreeInstance:
    center_x: float
    center_y: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    trunk_height: float
    trunk_radius: float
    lower_canopy_height: float
    lower_canopy_radius: float
    upper_canopy_height: float
    upper_canopy_radius: float


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect YOPO-style depth datasets from Isaac Lab random-forest maps.")
    parser.add_argument(
        "--output_root",
        "--output-root",
        "--save_path",
        dest="output_root",
        type=str,
        default=str(_DEFAULT_DATA_ROOT),
        help="Root directory that will contain timestamped dataset folders.",
    )
    parser.add_argument(
        "--dataset_timestamp",
        "--dataset-timestamp",
        dest="dataset_timestamp",
        type=str,
        default="",
        help="Optional explicit timestamp folder name. Defaults to YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--env_num",
        "--env-num",
        dest="env_num",
        type=int,
        default=10,
        help="Number of random maps to generate, matching YOPO's env_num concept.",
    )
    parser.add_argument(
        "--image_num",
        "--image-num",
        dest="image_num",
        type=int,
        default=10000,
        help="Number of depth images to collect per map.",
    )
    parser.add_argument(
        "--roll_range",
        "--roll-range",
        dest="roll_range",
        type=float,
        default=30.0,
        help="Maximum absolute roll angle in degrees for sampled camera poses.",
    )
    parser.add_argument(
        "--pitch_range",
        "--pitch-range",
        dest="pitch_range",
        type=float,
        default=30.0,
        help="Maximum absolute pitch angle in degrees for sampled camera poses.",
    )
    parser.add_argument(
        "--x_range",
        "--x-range",
        dest="x_range",
        type=float,
        default=40.0,
        help="Sampling width along world X, centered at the world origin.",
    )
    parser.add_argument(
        "--y_range",
        "--y-range",
        dest="y_range",
        type=float,
        default=40.0,
        help="Sampling width along world Y, centered at the world origin.",
    )
    parser.add_argument(
        "--z_range",
        "--z-range",
        dest="z_range",
        type=float,
        nargs=2,
        default=(0.5, 4.0),
        metavar=("Z_MIN", "Z_MAX"),
        help="Sampling range for the camera height in meters.",
    )
    parser.add_argument(
        "--safe_dist",
        "--safe-dist",
        dest="safe_dist",
        type=float,
        default=0.5,
        help="Minimum distance from the sampled camera position to the obstacle point cloud.",
    )
    parser.add_argument(
        "--map_seed",
        "--map-seed",
        dest="map_seed",
        type=int,
        default=0,
        help="Base seed used for per-map random pose sampling. Map k uses map_seed + k.",
    )
    parser.add_argument(
        "--max_sample_attempts",
        "--max-sample-attempts",
        dest="max_sample_attempts",
        type=int,
        default=5000,
        help="Maximum rejection-sampling attempts per collected image.",
    )
    parser.add_argument(
        "--camera_pitch_deg",
        "--camera-pitch-deg",
        dest="camera_pitch_deg",
        type=float,
        default=0.0,
        help="Fixed extra camera pitch applied after the sampled roll/pitch/yaw. Original YOPO uses 0 deg.",
    )
    parser.add_argument(
        "--pointcloud_surface_samples",
        "--pointcloud-surface-samples",
        dest="pointcloud_surface_samples",
        type=int,
        default=48,
        help="Approximate number of surface sample points generated per tree for PLY export and safe-distance checks.",
    )
    parser.add_argument(
        "--disable_pointcloud_ground",
        "--disable-pointcloud-ground",
        dest="disable_pointcloud_ground",
        action="store_true",
        default=False,
        help="Do not add sampled ground-plane points into the exported pointcloud PLY.",
    )
    parser.add_argument(
        "--pointcloud_ground_grid_size",
        "--pointcloud-ground-grid-size",
        dest="pointcloud_ground_grid_size",
        type=float,
        default=0.1,
        help="Grid spacing in meters for sampled ground-plane points written into the pointcloud PLY.",
    )
    parser.add_argument(
        "--pointcloud_ground_margin",
        "--pointcloud-ground-margin",
        dest="pointcloud_ground_margin",
        type=float,
        default=10.0,
        help="Extra XY margin added around the pose sampling window when generating ground-plane points.",
    )
    parser.add_argument(
        "--pointcloud_ground_z",
        "--pointcloud-ground-z",
        dest="pointcloud_ground_z",
        type=float,
        default=0.0,
        help="Ground-plane height used for sampled pointcloud floor points.",
    )
    parser.add_argument(
        "--disable_pointcloud",
        "--disable-pointcloud",
        dest="disable_pointcloud",
        action="store_true",
        default=False,
        help="Skip writing pointcloud-<map>.ply files.",
    )
    parser.add_argument(
        "--sim_dt",
        "--sim-dt",
        dest="sim_dt",
        type=float,
        default=0.01,
        help="Simulation dt used by Isaac Lab while refreshing the camera.",
    )
    parser.add_argument(
        "--sim_device",
        "--sim-device",
        dest="sim_device",
        type=str,
        default="cuda:0",
        help="Simulation device passed into Isaac Lab.",
    )
    parser.add_argument(
        "--world_path",
        "--world-path",
        dest="world_path",
        type=str,
        default="/World",
        help="USD world path used for the generated dataset stage.",
    )
    parser.add_argument(
        "--tiled_cam_prim_path",
        "--tiled-cam-prim-path",
        dest="tiled_cam_prim_path",
        type=str,
        default="/World/DatasetCamera/TiledCamera",
        help="USD prim path for the camera used to collect the dataset.",
    )
    parser.add_argument(
        "--tiled_cam_width",
        "--tiled-cam-width",
        dest="tiled_cam_width",
        type=int,
        default=160,
        help="Dataset camera width.",
    )
    parser.add_argument(
        "--tiled_cam_height",
        "--tiled-cam-height",
        dest="tiled_cam_height",
        type=int,
        default=96,
        help="Dataset camera height.",
    )
    parser.add_argument(
        "--tiled_cam_update_period",
        "--tiled-cam-update-period",
        dest="tiled_cam_update_period",
        type=float,
        default=0.0,
        help="Camera update period passed into TiledCameraCfg.",
    )
    parser.add_argument(
        "--tiled_cam_offset_pos",
        "--tiled-cam-offset-pos",
        dest="tiled_cam_offset_pos",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Static local camera offset. Usually left at zero for dataset collection.",
    )
    parser.add_argument(
        "--tiled_cam_offset_rot",
        "--tiled-cam-offset-rot",
        dest="tiled_cam_offset_rot",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
        help="Static local camera quaternion offset in (w x y z).",
    )
    parser.add_argument(
        "--tiled_cam_offset_convention",
        "--tiled-cam-offset-convention",
        dest="tiled_cam_offset_convention",
        choices=("world", "ros", "opengl"),
        default="world",
        help="Orientation convention used by the static camera offset.",
    )
    parser.add_argument(
        "--tiled_cam_focal_length",
        "--tiled-cam-focal-length",
        dest="tiled_cam_focal_length",
        type=float,
        default=24.0,
        help="Pinhole camera focal length.",
    )
    parser.add_argument(
        "--tiled_cam_focus_distance",
        "--tiled-cam-focus-distance",
        dest="tiled_cam_focus_distance",
        type=float,
        default=400.0,
        help="Pinhole camera focus distance.",
    )
    parser.add_argument(
        "--tiled_cam_horizontal_aperture",
        "--tiled-cam-horizontal-aperture",
        dest="tiled_cam_horizontal_aperture",
        type=float,
        default=20.955,
        help="Pinhole camera horizontal aperture.",
    )
    parser.add_argument(
        "--tiled_cam_depth_clip_near",
        "--tiled-cam-depth-clip-near",
        dest="tiled_cam_depth_clip_near",
        type=float,
        default=0.05,
        help="Near clipping plane for the depth camera.",
    )
    parser.add_argument(
        "--tiled_cam_depth_clip_far",
        "--tiled-cam-depth-clip-far",
        dest="tiled_cam_depth_clip_far",
        type=float,
        default=20.0,
        help="Far clipping plane for the depth camera.",
    )
    parser.add_argument(
        "--max_depth_dist",
        "--max-depth-dist",
        dest="max_depth_dist",
        type=float,
        default=20.0,
        help="Depth normalization distance used when saving 16-bit PNGs.",
    )
    parser.add_argument(
        "--camera_warmup_steps",
        "--camera-warmup-steps",
        dest="camera_warmup_steps",
        type=int,
        default=5,
        help="Number of render/update iterations used to warm the camera after creation or map swaps.",
    )
    parser.add_argument(
        "--pose_render_steps",
        "--pose-render-steps",
        dest="pose_render_steps",
        type=int,
        default=2,
        help="Number of render/update iterations after moving the camera before reading depth.",
    )

    add_random_forest_arguments(
        parser,
        default_size_x=60.0,
        default_size_y=60.0,
        default_tile_radius=0,
        default_clearance_radius=0.0,
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _resolve_dataset_dir(output_root: Path, requested_timestamp: str) -> Path:
    timestamp = requested_timestamp.strip() or time.strftime("%Y%m%d_%H%M%S")
    candidate = output_root / timestamp
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        candidate = output_root / f"{timestamp}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _serialize_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_jsonable(v) for v in value]
    return str(value)


def _quaternion_wxyz_from_euler_xyz_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ],
        dtype=np.float32,
    )


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = [float(v) for v in lhs]
    rw, rx, ry, rz = [float(v) for v in rhs]
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def _rotation_matrix_from_quaternion_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _tree_rotation_matrix(tree: TreeInstance) -> np.ndarray:
    quat = _quaternion_wxyz_from_euler_xyz_deg(tree.roll_deg, tree.pitch_deg, tree.yaw_deg)
    return _rotation_matrix_from_quaternion_wxyz(quat)


def _generate_tree_layout(cfg: RandomForestSceneCfg) -> list[TreeInstance]:
    layout: list[TreeInstance] = []
    for tile_x in range(-cfg.tile_radius, cfg.tile_radius + 1):
        for tile_y in range(-cfg.tile_radius, cfg.tile_radius + 1):
            tile_center_x = float(tile_x) * cfg.size_x
            tile_center_y = float(tile_y) * cfg.size_y
            rng = random.Random(_tile_seed(cfg.seed, tile_x, tile_y))
            positions = _generate_tile_positions(cfg, rng, tile_center_x=tile_center_x, tile_center_y=tile_center_y)
            for local_x, local_y in positions:
                scale_factor = rng.uniform(cfg.scale_min, cfg.scale_max)
                trunk_radius_scale = rng.uniform(cfg.trunk_radius_scale_min, cfg.trunk_radius_scale_max)
                lower_canopy_scale = rng.uniform(cfg.canopy_scale_min, cfg.canopy_scale_max)
                upper_canopy_scale = rng.uniform(cfg.canopy_scale_min, min(cfg.canopy_scale_max, lower_canopy_scale))
                layout.append(
                    TreeInstance(
                        center_x=tile_center_x + local_x,
                        center_y=tile_center_y + local_y,
                        roll_deg=rng.uniform(-cfg.tilt_deg_max, cfg.tilt_deg_max),
                        pitch_deg=rng.uniform(-cfg.tilt_deg_max, cfg.tilt_deg_max),
                        yaw_deg=rng.uniform(0.0, 360.0),
                        trunk_height=cfg.base_trunk_height * scale_factor,
                        trunk_radius=cfg.base_trunk_radius * scale_factor * trunk_radius_scale,
                        lower_canopy_height=cfg.base_lower_canopy_height * scale_factor * lower_canopy_scale,
                        lower_canopy_radius=cfg.base_lower_canopy_radius * scale_factor * lower_canopy_scale,
                        upper_canopy_height=cfg.base_upper_canopy_height * scale_factor * upper_canopy_scale,
                        upper_canopy_radius=cfg.base_upper_canopy_radius * scale_factor * upper_canopy_scale,
                    )
                )
    return layout


def _sample_cylinder_surface(radius: float, z_min: float, z_max: float, azimuth_count: int, z_count: int) -> np.ndarray:
    azimuths = np.linspace(0.0, 2.0 * math.pi, max(int(azimuth_count), 4), endpoint=False, dtype=np.float32)
    heights = np.linspace(float(z_min), float(z_max), max(int(z_count), 2), dtype=np.float32)
    aa, zz = np.meshgrid(azimuths, heights, indexing="xy")
    return np.stack((radius * np.cos(aa), radius * np.sin(aa), zz), axis=-1).reshape(-1, 3)


def _sample_cone_surface(
    radius: float,
    base_z: float,
    height: float,
    azimuth_count: int,
    height_count: int,
) -> np.ndarray:
    azimuths = np.linspace(0.0, 2.0 * math.pi, max(int(azimuth_count), 4), endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 1.0, max(int(height_count), 2), dtype=np.float32)
    aa, tt = np.meshgrid(azimuths, t, indexing="xy")
    rr = radius * (1.0 - tt)
    zz = base_z + height * tt
    surface = np.stack((rr * np.cos(aa), rr * np.sin(aa), zz), axis=-1).reshape(-1, 3)
    apex = np.array([[0.0, 0.0, base_z + height]], dtype=np.float32)
    base_ring = np.stack((radius * np.cos(azimuths), radius * np.sin(azimuths), np.full_like(azimuths, base_z)), axis=-1)
    return np.vstack((surface, apex, base_ring.astype(np.float32)))


def _sample_tree_surface_points(tree: TreeInstance, samples_per_tree: int) -> np.ndarray:
    samples_per_tree = max(int(samples_per_tree), 12)
    azimuth_count = max(6, int(math.sqrt(samples_per_tree * 1.5)))
    trunk_height_count = max(3, samples_per_tree // 24)
    canopy_height_count = max(3, samples_per_tree // 18)

    trunk_local = _sample_cylinder_surface(
        radius=tree.trunk_radius,
        z_min=0.0,
        z_max=tree.trunk_height,
        azimuth_count=azimuth_count,
        z_count=trunk_height_count,
    )
    lower_base_z = tree.trunk_height - 0.15 * tree.lower_canopy_height
    lower_local = _sample_cone_surface(
        radius=tree.lower_canopy_radius,
        base_z=lower_base_z,
        height=tree.lower_canopy_height,
        azimuth_count=azimuth_count,
        height_count=canopy_height_count,
    )
    upper_base_z = tree.trunk_height + 0.90 * tree.lower_canopy_height - 0.50 * tree.upper_canopy_height
    upper_local = _sample_cone_surface(
        radius=tree.upper_canopy_radius,
        base_z=upper_base_z,
        height=tree.upper_canopy_height,
        azimuth_count=max(4, azimuth_count // 2),
        height_count=max(2, canopy_height_count - 1),
    )

    local_points = np.vstack((trunk_local, lower_local, upper_local))
    rot = _tree_rotation_matrix(tree)
    translated = (rot @ local_points.T).T
    translated[:, 0] += tree.center_x
    translated[:, 1] += tree.center_y
    return translated.astype(np.float32)


def _build_map_pointcloud(layout: list[TreeInstance], samples_per_tree: int) -> np.ndarray:
    if not layout:
        return np.empty((0, 3), dtype=np.float32)
    point_sets = [_sample_tree_surface_points(tree, samples_per_tree=samples_per_tree) for tree in layout]
    return np.vstack(point_sets).astype(np.float32)


def _serialize_trunk_metadata(layout: list[TreeInstance]) -> list[dict[str, float]]:
    trunks: list[dict[str, float]] = []
    for tree in layout:
        trunks.append(
            {
                "x": float(tree.center_x),
                "y": float(tree.center_y),
                "z": 0.0,
                "roll_deg": float(tree.roll_deg),
                "pitch_deg": float(tree.pitch_deg),
                "yaw_deg": float(tree.yaw_deg),
                "radius": float(tree.trunk_radius),
                "height": float(tree.trunk_height),
            }
        )
    return trunks


def _build_ground_pointcloud(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_size: float,
    ground_z: float,
) -> np.ndarray:
    if x_max < x_min or y_max < y_min:
        return np.empty((0, 3), dtype=np.float32)

    step = max(float(grid_size), 1e-3)
    xs = np.arange(float(x_min), float(x_max) + 0.5 * step, step, dtype=np.float32)
    ys = np.arange(float(y_min), float(y_max) + 0.5 * step, step, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full_like(xx, float(ground_z), dtype=np.float32)
    return np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float32)


class _NearestSurfaceIndex:
    def __init__(self, points: np.ndarray):
        self._points = np.asarray(points, dtype=np.float32)
        self._tree = cKDTree(self._points) if cKDTree is not None and len(self._points) > 0 else None

    def distance(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float32).reshape(3)
        if self._tree is not None:
            dist, _ = self._tree.query(point, k=1)
            return float(dist)
        if len(self._points) == 0:
            return float("inf")
        delta = self._points - point[None, :]
        return float(np.sqrt(np.min(np.einsum("ij,ij->i", delta, delta))))


def _save_depth_as_16bit_png(depth_float: np.ndarray, max_depth_dist: float, path: Path) -> None:
    depth_scaled = np.nan_to_num(depth_float.astype(np.float32), nan=max_depth_dist, posinf=max_depth_dist, neginf=0.0)
    depth_scaled = np.clip(depth_scaled / max(max_depth_dist, 1e-6), 0.0, 1.0)
    depth_uint16 = np.round(depth_scaled * 65535.0).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        if not cv2.imwrite(str(path), depth_uint16):
            raise RuntimeError(f"Failed to write PNG: {path}")
        return
    if Image is None:
        raise RuntimeError("Neither cv2 nor PIL is available to write PNG files.")
    Image.fromarray(depth_uint16).save(path)


def _write_binary_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(points, dtype="<f4").reshape(-1, 3)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("wb") as ply_file:
        ply_file.write(header.encode("ascii"))
        vertices.tofile(ply_file)


def _resolve_map_output_paths(dataset_dir: Path, *, map_idx: int, map_count: int, write_pointcloud: bool) -> tuple[Path, Path, Path | None]:
    if int(map_count) <= 1:
        image_dir = dataset_dir / "img"
        pose_path = dataset_dir / "pose.csv"
        pointcloud_path = dataset_dir / "pointcloud.ply" if write_pointcloud else None
        return image_dir, pose_path, pointcloud_path

    image_dir = dataset_dir / f"img_{map_idx}"
    pose_path = dataset_dir / f"pose_{map_idx}.csv"
    pointcloud_path = dataset_dir / f"pointcloud_{map_idx}.ply" if write_pointcloud else None
    return image_dir, pose_path, pointcloud_path


def _resolve_depth_output(depth_camera: Any) -> np.ndarray:
    depth_tensor = depth_camera.data.output.get("depth")
    if depth_tensor is None:
        depth_tensor = depth_camera.data.output.get("distance_to_image_plane")
    if depth_tensor is None:
        raise RuntimeError("Depth camera did not produce a depth output tensor.")
    depth = depth_tensor[0, :, :, 0].detach().cpu().numpy().astype(np.float32)
    return depth


def _warm_camera(sim: Any, depth_camera: Any, *, steps: int, sim_dt: float) -> None:
    for _ in range(max(int(steps), 1)):
        sim.step()
        depth_camera.update(float(sim_dt), force_recompute=True)


def _import_camera_modules() -> tuple[Any, Any, Any]:
    _ensure_isaaclab_pythonpath()
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
    except ImportError as exc:
        raise SystemExit(
            "This script requires Isaac Lab camera modules (isaaclab package not found). "
            "Run it from the Isaac Lab environment."
        ) from exc
    return sim_utils, Camera, CameraCfg


def _add_depth_camera(
    args: argparse.Namespace,
    *,
    sim_utils: Any,
    Camera: Any,
    CameraCfg: Any,
    prim_path: str,
    log_name: str,
) -> Any:
    depth_camera_cfg = CameraCfg(
        prim_path=str(prim_path),
        update_period=float(args.tiled_cam_update_period),
        width=int(args.tiled_cam_width),
        height=int(args.tiled_cam_height),
        data_types=["depth"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args.tiled_cam_focal_length),
            focus_distance=float(args.tiled_cam_focus_distance),
            horizontal_aperture=float(args.tiled_cam_horizontal_aperture),
            clipping_range=(
                float(args.tiled_cam_depth_clip_near),
                float(args.tiled_cam_depth_clip_far),
            ),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in args.tiled_cam_offset_pos),
            rot=tuple(float(v) for v in args.tiled_cam_offset_rot),
            convention=str(args.tiled_cam_offset_convention),
        ),
    )
    depth_camera = Camera(depth_camera_cfg)
    print(
        f"{log_name} created:"
        f" prim={prim_path},"
        " data_types=['depth'],"
        f" resolution=({args.tiled_cam_width}x{args.tiled_cam_height})",
        flush=True,
    )
    return depth_camera


def _sample_pose(
    rng: np.random.Generator,
    surface_index: _NearestSurfaceIndex,
    *,
    x_range: float,
    y_range: float,
    z_min: float,
    z_max: float,
    roll_range: float,
    pitch_range: float,
    camera_pitch_deg: float,
    safe_dist: float,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(max(int(max_attempts), 1)):
        pos = np.array(
            [
                rng.uniform(-0.5 * x_range, 0.5 * x_range),
                rng.uniform(-0.5 * y_range, 0.5 * y_range),
                rng.uniform(z_min, z_max),
            ],
            dtype=np.float32,
        )
        if surface_index.distance(pos) < safe_dist:
            continue

        roll_deg = float(rng.normal(0.0, roll_range / 3.0))
        pitch_deg = float(rng.normal(0.0, pitch_range / 3.0))
        roll_deg = float(np.clip(roll_deg, -roll_range, roll_range))
        pitch_deg = float(np.clip(pitch_deg, -pitch_range, pitch_range))
        yaw_deg = float(rng.uniform(0.0, 360.0))

        quat_body = _quaternion_wxyz_from_euler_xyz_deg(roll_deg, pitch_deg, yaw_deg)
        quat_mount = _quaternion_wxyz_from_euler_xyz_deg(0.0, camera_pitch_deg, 0.0)
        quat_world = _quat_mul_wxyz(quat_body, quat_mount)
        return pos, quat_world

    raise RuntimeError(
        "Failed to sample a collision-free pose. "
        "Reduce --safe_dist or widen the sampling range / z-range."
    )


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    if args.env_num <= 0:
        parser.error("--env_num must be > 0.")
    if args.image_num <= 0:
        parser.error("--image_num must be > 0.")
    if args.x_range <= 0.0 or args.y_range <= 0.0:
        parser.error("--x_range and --y_range must be > 0.")
    if args.z_range[0] >= args.z_range[1]:
        parser.error("--z_range requires Z_MIN < Z_MAX.")
    if args.safe_dist < 0.0:
        parser.error("--safe_dist must be >= 0.")
    if args.pointcloud_ground_grid_size <= 0.0:
        parser.error("--pointcloud_ground_grid_size must be > 0.")
    if args.pointcloud_ground_margin < 0.0:
        parser.error("--pointcloud_ground_margin must be >= 0.")
    if args.tiled_cam_width <= 0 or args.tiled_cam_height <= 0:
        parser.error("--tiled_cam_width and --tiled_cam_height must be positive.")
    if args.max_depth_dist <= 0.0:
        parser.error("--max_depth_dist must be > 0.")
    if not args.world_path.startswith("/"):
        parser.error("--world_path must be an absolute USD path.")
    if not args.tiled_cam_prim_path.startswith("/"):
        parser.error("--tiled_cam_prim_path must be an absolute USD path.")

    output_root = Path(args.output_root).expanduser()
    dataset_dir = _resolve_dataset_dir(output_root, args.dataset_timestamp)
    dataset_dir.mkdir(parents=True, exist_ok=False)
    print(f"Dataset output directory: {dataset_dir}", flush=True)

    launcher = AppLauncher(
        headless=bool(getattr(args, "headless", False)),
        enable_cameras=True,
        fast_shutdown=bool(getattr(args, "headless", False)),
    )
    simulation_app = launcher.app
    sim = None

    try:
        _ensure_pxr_imported()
        sim_utils, Camera, CameraCfg = _import_camera_modules()

        initialize_scene_from_editor(
            sim_utils=sim_utils,
            world_path=str(args.world_path),
            add_lights=True,
            add_ground=True,
            create_new_stage=True,
        )

        sim_cfg = sim_utils.SimulationCfg(dt=float(args.sim_dt), device=str(args.sim_device))
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.cfg.add_ground_plane = False

        depth_camera = _add_depth_camera(
            args,
            sim_utils=sim_utils,
            Camera=Camera,
            CameraCfg=CameraCfg,
            prim_path=_derive_depth_tiled_camera_prim_path(args.tiled_cam_prim_path),
            log_name="Dataset depth camera",
        )
        sim_utils.update_stage()
        sim.reset()
        _warm_camera(sim, depth_camera, steps=int(args.camera_warmup_steps), sim_dt=float(args.sim_dt))

        forest_cfg = build_random_forest_cfg_from_args(args)

        metadata = {
            "dataset_dir": str(dataset_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "collector": "collect_yopo_dataset.py",
            "args": _serialize_jsonable(vars(args)),
            "maps": [],
        }

        total_images = args.env_num * args.image_num
        collected_images = 0

        for map_idx in range(args.env_num):
            per_map_cfg = RandomForestSceneCfg(**asdict(forest_cfg))
            per_map_cfg.seed = int(forest_cfg.seed) + map_idx
            forest_summary = add_random_forest_scene(sim_utils=sim_utils, cfg=per_map_cfg)
            sim_utils.update_stage()
            _warm_camera(
                sim,
                depth_camera,
                steps=max(2, int(args.camera_warmup_steps)),
                sim_dt=float(args.sim_dt),
            )

            tree_layout = _generate_tree_layout(per_map_cfg)
            tree_points = _build_map_pointcloud(tree_layout, samples_per_tree=int(args.pointcloud_surface_samples))
            ground_points = np.empty((0, 3), dtype=np.float32)
            if not bool(args.disable_pointcloud_ground):
                ground_margin = float(args.pointcloud_ground_margin)
                ground_points = _build_ground_pointcloud(
                    x_min=-0.5 * float(args.x_range) - ground_margin,
                    x_max=0.5 * float(args.x_range) + ground_margin,
                    y_min=-0.5 * float(args.y_range) - ground_margin,
                    y_max=0.5 * float(args.y_range) + ground_margin,
                    grid_size=float(args.pointcloud_ground_grid_size),
                    ground_z=float(args.pointcloud_ground_z),
                )

            if len(tree_points) > 0 and len(ground_points) > 0:
                map_points = np.vstack((tree_points, ground_points)).astype(np.float32)
            elif len(tree_points) > 0:
                map_points = tree_points
            else:
                map_points = ground_points
            surface_index = _NearestSurfaceIndex(map_points)

            image_dir, pose_path, pointcloud_path = _resolve_map_output_paths(
                dataset_dir,
                map_idx=map_idx,
                map_count=int(args.env_num),
                write_pointcloud=not args.disable_pointcloud,
            )

            if pointcloud_path is not None:
                _write_binary_ply(pointcloud_path, map_points)

            image_dir.mkdir(parents=True, exist_ok=True)

            map_rng = np.random.default_rng(int(args.map_seed) + map_idx)
            z_min, z_max = float(args.z_range[0]), float(args.z_range[1])

            with pose_path.open("w", encoding="utf-8", newline="") as pose_file:
                pose_file.write("px,py,pz,qw,qx,qy,qz\n")

                for image_idx in range(args.image_num):
                    position, quat_world = _sample_pose(
                        map_rng,
                        surface_index,
                        x_range=float(args.x_range),
                        y_range=float(args.y_range),
                        z_min=z_min,
                        z_max=z_max,
                        roll_range=float(args.roll_range),
                        pitch_range=float(args.pitch_range),
                        camera_pitch_deg=float(args.camera_pitch_deg),
                        safe_dist=float(args.safe_dist),
                        max_attempts=int(args.max_sample_attempts),
                    )

                    depth_camera.set_world_poses(
                        positions=position.reshape(1, 3),
                        orientations=quat_world.reshape(1, 4),
                        convention="world",
                    )
                    _warm_camera(
                        sim,
                        depth_camera,
                        steps=max(1, int(args.pose_render_steps)),
                        sim_dt=float(args.sim_dt),
                    )
                    depth = _resolve_depth_output(depth_camera)

                    image_path = image_dir / f"img_{image_idx}.png"
                    _save_depth_as_16bit_png(depth, max_depth_dist=float(args.max_depth_dist), path=image_path)
                    pose_file.write(
                        f"{position[0]:.6f},{position[1]:.6f},{position[2]:.6f},"
                        f"{quat_world[0]:.6f},{quat_world[1]:.6f},{quat_world[2]:.6f},{quat_world[3]:.6f}\n"
                    )

                    collected_images += 1
                    if collected_images == total_images or collected_images % 100 == 0:
                        print(
                            f"Collected {collected_images}/{total_images} images "
                            f"(map {map_idx + 1}/{args.env_num}, sample {image_idx + 1}/{args.image_num})",
                            flush=True,
                        )

            metadata["maps"].append(
                {
                    "map_idx": map_idx,
                    "forest_summary": _serialize_jsonable(forest_summary),
                    "tree_count": len(tree_layout),
                    "trunks": _serialize_trunk_metadata(tree_layout),
                    "tree_point_count": int(tree_points.shape[0]),
                    "ground_point_count": int(ground_points.shape[0]),
                    "point_count": int(map_points.shape[0]),
                    "image_dir": str(image_dir),
                    "pose_csv": str(pose_path),
                    "pointcloud_ply": str(pointcloud_path) if pointcloud_path is not None else None,
                }
            )

            metadata_path = dataset_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(
                f"Completed map {map_idx + 1}/{args.env_num}: "
                f"trees={len(tree_layout)}, tree_points={tree_points.shape[0]}, "
                f"ground_points={ground_points.shape[0]}, total_points={map_points.shape[0]}, "
                f"pose_csv={pose_path.name}",
                flush=True,
            )

        print(f"Dataset collection finished: {dataset_dir}", flush=True)
        if getattr(args, "headless", False):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return 0
    finally:
        if sim is not None:
            with contextlib.suppress(Exception):
                sim.stop()
                sim.clear_all_callbacks()
                sim.clear_instance()
        with contextlib.suppress(Exception):
            if getattr(args, "headless", False):
                simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
            else:
                simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
