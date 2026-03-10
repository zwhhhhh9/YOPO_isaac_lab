#!/usr/bin/env python3
"""
IsaacLab runner that streams depth/odometry via ROS 2.
Fix: Delays rclpy imports until AFTER Isaac Sim is initialized to avoid malloc corruption.
Fix: Added missing publishers and methods for flatness/control debug.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shlex
import socket
import struct
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


_CSV_LOG_DIR = _project_root() / "yopo_drone" / "logs"
_CSV_DEFAULT_DIR_ALIASES = (
    Path("."),
    Path("yopo_drone/data"),
    Path("yopo_drone/logs"),
)
_RUN_LOG_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")


def _normalize_csv_log_path(path_str: str, *, default_filename: str | None = None) -> str:
    raw_path = str(path_str).strip()
    if not raw_path:
        if default_filename is None:
            return ""
        return str((_CSV_LOG_DIR / default_filename).resolve())

    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    if path.parent in _CSV_DEFAULT_DIR_ALIASES:
        return str((_CSV_LOG_DIR / path.name).resolve())
    return str((_project_root() / path).resolve())


def _default_hover_telemetry_filename() -> str:
    return f"hvoer_{_RUN_LOG_TIMESTAMP}.csv"


def _ensure_isaaclab_pythonpath() -> None:
    source_roots = (
        _project_root() / "source",
        _project_root() / "env_tools" / "docker" / "isaaclab" / "IsaacLab" / "source",
    )
    for source_root in source_roots:
        candidates = (
            source_root / "isaaclab",
            source_root / "isaaclab_assets",
            source_root / "isaaclab_mimic",
            source_root / "isaaclab_rl",
            source_root / "isaaclab_tasks",
        )
        for package_root in candidates:
            package_root_str = str(package_root)
            if package_root.is_dir() and package_root_str not in sys.path:
                sys.path.insert(0, package_root_str)


try:
    from isaaclab.app import AppLauncher
except ImportError:
    _ensure_isaaclab_pythonpath()
    from isaaclab.app import AppLauncher

try:
    from yopo_drone.utils.px4_controller import PX4QuadrotorController
except ImportError:
    from e2e_drone.utils.px4_controller import PX4QuadrotorController

from yopo_drone.utils.robot_model import (
    DEFAULT_PX4_PARAMS,
    DEFAULT_ROBOT_URDF,
    load_px4_robot_model,
)


LEGACY_CTRL_MASS = 0.5
LEGACY_CTRL_INERTIA = [4.8847e-4, 1.0395e-3, 1.21702e-3]
LEGACY_CTRL_ARM_LENGTH = 0.1827
LEGACY_CTRL_MOTOR_THRUST_MAX = 4.5
LEGACY_CTRL_THRUST_RATIO = 0.6
LEGACY_CTRL_KAPPA = 0.015

LEGACY_POS_KP = [2.0, 2.0, 2.0]
LEGACY_VEL_KP = [8.0, 8.0, 8.0]
LEGACY_ATTITUDE_FB_KP = [20.0, 20.0, 20.0]
LEGACY_MAX_BODYRATE_FB = 4.0
LEGACY_MAX_ANGLE_DEG = 30.0
LEGACY_MIN_COLLECTIVE_ACC = 3.0
LEGACY_ATT_P_GAIN = [4.5, 4.5, 2.0]
LEGACY_ATT_RATE_LIMIT = [2.5, 2.5, 2.0]
LEGACY_ATT_YAW_WEIGHT = 0.4
LEGACY_RATE_P_GAIN = [0.02, 0.03, 0.015]
LEGACY_RATE_I_GAIN = [0.02, 0.02, 0.015]
LEGACY_RATE_D_GAIN = [0.0015, 0.0020, 0.0008]
LEGACY_RATE_K_GAIN = [1.0, 1.0, 1.0]
LEGACY_RATE_INT_LIMIT = [0.2, 0.2, 0.15]
LEGACY_ACCEL_FILTER_COEF = 0.5


class DepthSharedMemoryWriter:
    """Writes depth frames into a named shared memory block."""

    _HEADER = struct.Struct("<IId")  # width, height, timestamp

    def __init__(self, name: str) -> None:
        self._name = name
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._size = 0
        self._owns_segment = False

    def write(self, depth: np.ndarray, timestamp: float) -> None:
        if depth is None:
            return
        height, width = depth.shape
        depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        payload_size = width * height * 2
        total_size = self._HEADER.size + payload_size
        self._ensure_segment(total_size)
        if self._shm is None:
            return
        buf = self._shm.buf
        self._HEADER.pack_into(buf, 0, width, height, timestamp)
        start = self._HEADER.size
        mv = memoryview(buf)[start : start + payload_size]
        mv[:] = depth_mm.tobytes()

    def _ensure_segment(self, required_size: int) -> None:
        if self._shm is not None and self._size >= required_size:
            return
        self.close()
        try:
            self._shm = shared_memory.SharedMemory(name=self._name, create=True, size=required_size)
            self._size = required_size
            self._owns_segment = True
            return
        except FileExistsError:
            pass
        existing = shared_memory.SharedMemory(name=self._name, create=False)
        if existing.size < required_size:
            existing.close()
            raise RuntimeError(
                f"Existing shared memory '{self._name}' is too small ({existing.size} < {required_size})."
            )
        self._shm = existing
        self._size = existing.size
        self._owns_segment = False

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            if self._owns_segment:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
            self._shm = None
        self._owns_segment = False
        self._size = 0

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 streamer for IsaacLab quadrotor environments.")
    parser.add_argument("--task", type=str, default="point_ctrl_single_ego", help="IsaacLab task to load.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--sim_dt", type=float, default=0.01, help="Simulation time step.")
    parser.add_argument("--decimation", type=int, default=1, help="Simulation decimation.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric interface.")
    parser.add_argument("--drone_id", type=int, default=0, help="Drone ID for multi-agent tasks.")
    parser.add_argument("--frame_id", type=str, default="world", help="Frame identifier stamped into ROS messages.")
    parser.add_argument("--depth_topic", type=str, default="/drone_0_depth", help="Depth image topic.")
    parser.add_argument("--depth_shm_name", type=str, default="depth_image_shm", help="Shared memory name for depth frames.")
    parser.add_argument("--disable_depth_shm", action="store_true", default=False, help="Disable writing depth frames to shared memory.")
    parser.add_argument("--odom_topic", type=str, default="/drone_0_odometry", help="Odometry topic.")
    parser.add_argument("--cmd_topic", type=str, default="/drone_0_planning/pos_cmd", help="Position command topic.")
    parser.add_argument("--ctrl_attitude_des_topic", type=str, default="/drone_0_ctrl/attitude_des", help="Desired attitude topic.")
    parser.add_argument("--ctrl_attitude_real_topic", type=str, default="/drone_0_ctrl/attitude_real", help="Current attitude topic.")
    parser.add_argument("--ctrl_bodyrate_des_topic", type=str, default="/drone_0_ctrl/bodyrate_des", help="Desired body-rate topic.")
    parser.add_argument("--ctrl_bodyrate_real_topic", type=str, default="/drone_0_ctrl/bodyrate_real", help="Current body-rate topic.")
    parser.add_argument("--flatness_att_topic", type=str, default="/debug_flatness/attitude", help="Flatness attitude topic.")
    parser.add_argument("--flatness_rate_topic", type=str, default="/debug_flatness/bodyrate", help="Flatness bodyrate topic.")
    parser.add_argument("--flatness_thrust_topic", type=str, default="/debug_flatness/thrust", help="Flatness thrust topic.")
    parser.add_argument("--mass", type=float, default=0.5, help="Vehicle mass for trajectory conversion.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity value for trajectory conversion.")
    parser.add_argument("--ctrl-model-urdf", type=str, default=DEFAULT_ROBOT_URDF, help="URDF used to derive controller rigid-body parameters.")
    parser.add_argument("--ctrl-model-params", type=str, default=DEFAULT_PX4_PARAMS, help="JSON file with PX4 defaults for the selected URDF model.")
    parser.add_argument("--ctrl_mass", type=float, default=LEGACY_CTRL_MASS, help="Controller model mass [kg].")
    parser.add_argument(
        "--ctrl_inertia",
        type=float,
        nargs=3,
        default=LEGACY_CTRL_INERTIA,
        help="Controller model inertia [Ixx Iyy Izz] in kg*m^2.",
    )
    parser.add_argument("--ctrl_arm_length", type=float, default=LEGACY_CTRL_ARM_LENGTH, help="Controller model motor arm length [m].")
    parser.add_argument("--ctrl_motor_thrust_max", type=float, default=LEGACY_CTRL_MOTOR_THRUST_MAX, help="Estimated per-motor max thrust [N].")
    parser.add_argument(
        "--ctrl_thrust_ratio",
        type=float,
        default=LEGACY_CTRL_THRUST_RATIO,
        help="Available thrust ratio for differential control.",
    )
    parser.add_argument("--ctrl_kappa", type=float, default=LEGACY_CTRL_KAPPA, help="Yaw moment ratio (kappa).")
    parser.add_argument("--pos_kp", type=float, nargs=3, default=LEGACY_POS_KP, help="Outer-loop position gains [x y z].")
    parser.add_argument(
        "--vel_kp",
        type=float,
        nargs=3,
        default=LEGACY_VEL_KP,
        help="Velocity gains that convert velocity error into acceleration command [x y z].",
    )
    parser.add_argument(
        "--attitude_fb_kp",
        type=float,
        nargs=3,
        default=LEGACY_ATTITUDE_FB_KP,
        help="Quaternion feedback gains added to flatness body-rate feed-forward [roll pitch yaw].",
    )
    parser.add_argument("--max_bodyrate_fb", type=float, default=LEGACY_MAX_BODYRATE_FB, help="Max feedback body-rate magnitude [rad/s].")
    parser.add_argument("--max_angle_deg", type=float, default=LEGACY_MAX_ANGLE_DEG, help="Max commanded roll/pitch angle [deg].")
    parser.add_argument(
        "--min_collective_acc",
        type=float,
        default=LEGACY_MIN_COLLECTIVE_ACC,
        help="Minimum collective acceleration magnitude enforced by the flatness mapping [m/s^2].",
    )
    parser.add_argument(
        "--att_p_gain",
        type=float,
        nargs=3,
        default=LEGACY_ATT_P_GAIN,
        help="PX4 attitude P gains [roll pitch yaw].",
    )
    parser.add_argument(
        "--att_rate_limit",
        type=float,
        nargs=3,
        default=LEGACY_ATT_RATE_LIMIT,
        help="PX4 attitude-controller rate limits [roll pitch yaw] in rad/s.",
    )
    parser.add_argument("--att_yaw_weight", type=float, default=LEGACY_ATT_YAW_WEIGHT, help="PX4 attitude yaw priority weight [0, 1].")
    parser.add_argument(
        "--rate_p_gain",
        type=float,
        nargs=3,
        default=LEGACY_RATE_P_GAIN,
        help="PX4 rate-controller P gains [roll pitch yaw].",
    )
    parser.add_argument(
        "--rate_i_gain",
        type=float,
        nargs=3,
        default=LEGACY_RATE_I_GAIN,
        help="PX4 rate-controller I gains [roll pitch yaw].",
    )
    parser.add_argument(
        "--rate_d_gain",
        type=float,
        nargs=3,
        default=LEGACY_RATE_D_GAIN,
        help="PX4 rate-controller D gains [roll pitch yaw].",
    )
    parser.add_argument(
        "--rate_k_gain",
        type=float,
        nargs=3,
        default=LEGACY_RATE_K_GAIN,
        help="PX4 rate-controller K gains [roll pitch yaw].",
    )
    parser.add_argument(
        "--rate_int_limit",
        type=float,
        nargs=3,
        default=LEGACY_RATE_INT_LIMIT,
        help="PX4 rate-controller integral limits [roll pitch yaw].",
    )
    parser.add_argument(
        "--accel_filter_coef",
        type=float,
        default=LEGACY_ACCEL_FILTER_COEF,
        help="Angular-acceleration low-pass coefficient used by the rate D-term [0, 1].",
    )
    parser.add_argument(
        "--reset_log_path",
        type=str,
        default="./yopo_drone/logs/ego.csv",
        help="CSV path for reset stats. Relative default-style CSV paths are saved under yopo_drone/logs/.",
    )
    parser.add_argument("--reset_log_count", type=int, default=26, help="Number of resets to log before stopping.")
    parser.add_argument(
        "--startup_hover_settle_steps",
        type=int,
        default=400,
        help="Maximum number of closed-loop hover warm-up steps to run before starting telemetry and command handling.",
    )
    parser.add_argument(
        "--telemetry_log_path",
        type=str,
        default=_default_hover_telemetry_filename(),
        help=(
            "CSV path for per-step hover telemetry. Defaults to a per-run file named "
            "hvoer_YYYYMMDD_HHMMSS.csv under yopo_drone/logs/."
        ),
    )
    parser.add_argument(
        "--disable_rotor_spin_visual",
        action="store_true",
        default=False,
        help="Disable GUI-only rotor spin animation driven by controller motor speeds.",
    )
    parser.add_argument(
        "--rotor_spin_visual_scale",
        type=float,
        default=0.12,
        help="Visual-only multiplier applied to rotor angular speed in GUI mode.",
    )
    parser.add_argument(
        "--disable_tiled_camera",
        "--disable-tiled-camera",
        dest="disable_tiled_camera",
        action="store_true",
        default=False,
        help="Disable the tiled camera sensor used for depth preview and shared-memory export.",
    )
    parser.add_argument(
        "--tiled_cam_prim_path",
        "--tiled-cam-prim-path",
        dest="tiled_cam_prim_path",
        type=str,
        default="{ENV_REGEX_NS}/Robot/base_link/TiledCamera",
        help="USD path for the tiled camera. Supports {ENV_REGEX_NS} when attached to environment robots.",
    )
    parser.add_argument(
        "--tiled_cam_width",
        "--tiled-cam-width",
        dest="tiled_cam_width",
        type=int,
        default=160,
        help="Tiled camera width.",
    )
    parser.add_argument(
        "--tiled_cam_height",
        "--tiled-cam-height",
        dest="tiled_cam_height",
        type=int,
        default=96,
        help="Tiled camera height.",
    )
    parser.add_argument(
        "--tiled_cam_update_period",
        "--tiled-cam-update-period",
        dest="tiled_cam_update_period",
        type=float,
        default=0.0,
        help="Tiled camera update period in seconds.",
    )
    parser.add_argument(
        "--tiled_cam_offset_pos",
        "--tiled-cam-offset-pos",
        dest="tiled_cam_offset_pos",
        type=float,
        nargs=3,
        default=(0.20, 0.0, 0.0),
        help="Camera offset position relative to the robot frame.",
    )
    parser.add_argument(
        "--tiled_cam_offset_rot",
        "--tiled-cam-offset-rot",
        dest="tiled_cam_offset_rot",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
        help="Camera offset rotation quaternion (w x y z).",
    )
    parser.add_argument(
        "--tiled_cam_offset_convention",
        "--tiled-cam-offset-convention",
        dest="tiled_cam_offset_convention",
        type=str,
        choices=("world", "ros", "opengl"),
        default="world",
        help="Orientation convention used by the tiled camera offset.",
    )
    parser.add_argument(
        "--tiled_cam_focal_length",
        "--tiled-cam-focal-length",
        dest="tiled_cam_focal_length",
        type=float,
        default=24.0,
        help="Tiled camera focal length.",
    )
    parser.add_argument(
        "--tiled_cam_focus_distance",
        "--tiled-cam-focus-distance",
        dest="tiled_cam_focus_distance",
        type=float,
        default=400.0,
        help="Tiled camera focus distance.",
    )
    parser.add_argument(
        "--tiled_cam_horizontal_aperture",
        "--tiled-cam-horizontal-aperture",
        dest="tiled_cam_horizontal_aperture",
        type=float,
        default=20.955,
        help="Tiled camera horizontal aperture.",
    )
    parser.add_argument(
        "--tiled_cam_clip_near",
        "--tiled-cam-clip-near",
        dest="tiled_cam_clip_near",
        type=float,
        default=0.05,
        help="Near clipping plane for the tiled camera.",
    )
    parser.add_argument(
        "--tiled_cam_clip_far",
        "--tiled-cam-clip-far",
        dest="tiled_cam_clip_far",
        type=float,
        default=200.0,
        help="Far clipping plane for the tiled camera.",
    )
    parser.add_argument(
        "--tiled_cam_warmup_steps",
        "--tiled-cam-warmup-steps",
        dest="tiled_cam_warmup_steps",
        type=int,
        default=6,
        help="Number of render-only warmup iterations before reading tiled camera outputs.",
    )
    parser.add_argument(
        "--disable_tiled_camera_inset",
        "--disable-tiled-camera-inset",
        dest="disable_tiled_camera_inset",
        action="store_true",
        default=False,
        help="Disable the GUI inset viewport that follows the tiled camera.",
    )
    parser.add_argument(
        "--tiled_cam_inset_window_name",
        "--tiled-cam-inset-window-name",
        dest="tiled_cam_inset_window_name",
        type=str,
        default="Tiled Camera",
        help="Window title for the tiled camera GUI inset.",
    )
    parser.add_argument(
        "--tiled_cam_inset_width",
        "--tiled-cam-inset-width",
        dest="tiled_cam_inset_width",
        type=int,
        default=360,
        help="Width of the tiled camera GUI inset.",
    )
    parser.add_argument(
        "--tiled_cam_inset_height",
        "--tiled-cam-inset-height",
        dest="tiled_cam_inset_height",
        type=int,
        default=240,
        help="Height of the tiled camera GUI inset.",
    )
    parser.add_argument(
        "--tiled_cam_inset_pos_x",
        "--tiled-cam-inset-pos-x",
        dest="tiled_cam_inset_pos_x",
        type=int,
        default=60,
        help="Fallback inset X position when Render Settings window is unavailable.",
    )
    parser.add_argument(
        "--tiled_cam_inset_pos_y",
        "--tiled-cam-inset-pos-y",
        dest="tiled_cam_inset_pos_y",
        type=int,
        default=120,
        help="Fallback inset Y position when Render Settings window is unavailable.",
    )
    parser.add_argument("--disable_env_editor_scene_init", action="store_true", default=False, help="Do not initialize the stage with drone_env_editor-style world primitives before launching the planner/control environment.")
    parser.add_argument("--env_editor_world_path", type=str, default="/World", help="USD world path used when applying drone_env_editor scene initialization.")
    parser.add_argument("--disable_env_editor_lights", action="store_true", default=False, help="Skip adding drone_env_editor dome light during stage initialization.")
    parser.add_argument("--disable_env_editor_ground", action="store_true", default=False, help="Skip adding drone_env_editor ground plane during stage initialization.")
    parser.add_argument("--disable_ros2_bridge", action="store_true", default=False, help="Disable UDP sidecar ROS 2 bridge when direct rclpy is unavailable.")
    parser.add_argument("--ros2_distro", type=str, default="jazzy", help="System ROS 2 distro used by the sidecar bridge.")
    parser.add_argument("--ros2_workspace", type=str, default="/workspace/isaaclab/ros2_ws", help="ROS 2 workspace path inside the container.")
    parser.add_argument("--ros2_bridge_python", type=str, default="/usr/bin/python3", help="Python executable for the ROS 2 sidecar bridge.")
    parser.add_argument("--ros2_cmd_host", type=str, default="127.0.0.1", help="UDP host for receiving PositionCommand from the ROS 2 sidecar bridge.")
    parser.add_argument("--ros2_cmd_port", type=int, default=15000, help="UDP port for receiving PositionCommand from the ROS 2 sidecar bridge.")
    parser.add_argument("--ros2_state_host", type=str, default="127.0.0.1", help="UDP host for sending odom/reset/goal state to the ROS 2 sidecar bridge.")
    parser.add_argument("--ros2_state_port", type=int, default=15001, help="UDP port for sending odom/reset/goal state to the ROS 2 sidecar bridge.")
    
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if not getattr(args, "disable_tiled_camera", False):
        args.enable_cameras = True
    sys.argv = [sys.argv[0]] + hydra_args
    return args

def main() -> None:
    # 1. 解析参数
    args_cli = _parse_arguments()
    
    # 2. 启动 Isaac Sim 应用 (必须是第一步！)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # =================================================================================
    # 关键：只有在 simulation_app 启动后，才能导入 ROS 2 库
    # =================================================================================
    ros_enabled = False
    try:
        import rclpy
        from rclpy.node import Node as RosNode
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import Vector3Stamped
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import Image
        from std_msgs.msg import Bool
        from quadrotor_msgs.msg import PositionCommand
        ros_enabled = True
    except ImportError as exc:
        rclpy = None
        RosNode = object
        Odometry = None
        Vector3Stamped = None
        PoseStamped = None
        Image = None
        Bool = None
        PositionCommand = None
        print(f"[Warn] ROS 2 bridge disabled: {exc}")

    class _DummyLogger:
        def info(self, msg: str) -> None:
            print(f"[Info] {msg}")

        def warning(self, msg: str) -> None:
            print(f"[Warn] {msg}")

        warn = warning

        def error(self, msg: str) -> None:
            print(f"[Error] {msg}")

    class _DummyPublisher:
        def publish(self, _msg) -> None:
            return

    class _DummyNode:
        def __init__(self, _name: str) -> None:
            self._dummy_logger = _DummyLogger()

        def create_publisher(self, *_args, **_kwargs):
            return _DummyPublisher()

        def create_subscription(self, *_args, **_kwargs):
            return None

        def destroy_node(self) -> None:
            return

        def get_logger(self) -> _DummyLogger:
            return self._dummy_logger

    LEGACY_CTRL_INERTIA_NP = np.array(LEGACY_CTRL_INERTIA, dtype=np.float32)

    NodeBase = RosNode if ros_enabled else _DummyNode

    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    try:
        from yopo_drone import tasks
    except ImportError:
        from e2e_drone import tasks
    import gymnasium as gym

    def _tiled_camera_enabled() -> bool:
        return not bool(getattr(args_cli, "disable_tiled_camera", False))

    def _resolve_eval_tiled_camera_paths(unwrapped) -> tuple[str, str]:
        prim_path = str(args_cli.tiled_cam_prim_path).strip()
        if not prim_path:
            raise ValueError("--tiled_cam_prim_path must not be empty.")

        scene = getattr(unwrapped, "scene", None)
        env_regex_ns = str(getattr(scene, "env_regex_ns", "") or "").rstrip("/")
        env_prim_paths = getattr(scene, "env_prim_paths", None) or []
        env0_prim_path = str(env_prim_paths[0]).rstrip("/") if env_prim_paths else ""

        if prim_path.startswith("/World/Robot") and env_regex_ns:
            prim_path = prim_path.replace("/World/Robot", f"{env_regex_ns}/Robot", 1)
        elif "{ENV_REGEX_NS}" in prim_path:
            if not env_regex_ns:
                raise RuntimeError("Unable to resolve {ENV_REGEX_NS} for tiled camera prim path.")
            prim_path = prim_path.replace("{ENV_REGEX_NS}", env_regex_ns)

        if not prim_path.startswith("/"):
            raise ValueError(
                f"Resolved tiled camera prim path must be absolute, got '{prim_path}'."
            )

        view_path = prim_path
        if env_regex_ns and env0_prim_path and env_regex_ns in view_path:
            view_path = view_path.replace(env_regex_ns, env0_prim_path, 1)

        return prim_path, view_path

    def _warm_up_eval_tiled_camera(unwrapped) -> None:
        tiled_camera = getattr(unwrapped, "_tiled_camera", None)
        if tiled_camera is None:
            return

        if (not getattr(tiled_camera, "_is_initialized", False)) or (not hasattr(tiled_camera, "_timestamp")):
            tiled_camera._initialize_impl()
            tiled_camera._is_initialized = True
            with contextlib.suppress(Exception):
                tiled_camera.reset()

        for _ in range(max(int(args_cli.tiled_cam_warmup_steps), 1)):
            unwrapped.sim.render()
            tiled_camera.update(0.0, force_recompute=True)

        rgb = tiled_camera.data.output.get("rgb")
        depth = tiled_camera.data.output.get("depth")
        if depth is None:
            depth = tiled_camera.data.output.get("distance_to_image_plane")
        rgb_shape = tuple(rgb.shape) if rgb is not None else None
        depth_shape = tuple(depth.shape) if depth is not None else None
        print(f"[Info] Tiled camera data ready: rgb_shape={rgb_shape}, depth_shape={depth_shape}")

    def _maybe_create_eval_tiled_camera(env) -> None:
        unwrapped = env.unwrapped
        unwrapped._tiled_camera = None
        unwrapped._tiled_camera_inset_window = None
        unwrapped._tiled_camera_view_prim_path = None
        if not _tiled_camera_enabled():
            return

        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCamera, TiledCameraCfg

        try:
            from yopo_drone.env.drone_env_editor import _add_tiled_camera, _attach_tiled_camera_inset
        except ImportError:
            from e2e_drone.env.drone_env_editor import _add_tiled_camera, _attach_tiled_camera_inset

        sensor_prim_path, view_prim_path = _resolve_eval_tiled_camera_paths(unwrapped)
        tiled_camera_args = argparse.Namespace(**vars(args_cli))
        tiled_camera_args.tiled_cam_prim_path = sensor_prim_path

        tiled_camera = _add_tiled_camera(
            tiled_camera_args,
            sim_utils=sim_utils,
            TiledCamera=TiledCamera,
            TiledCameraCfg=TiledCameraCfg,
        )
        sim_utils.update_stage()

        unwrapped._tiled_camera = tiled_camera
        unwrapped._tiled_camera_view_prim_path = view_prim_path

        _warm_up_eval_tiled_camera(unwrapped)

        inset_window = None
        if not bool(getattr(args_cli, "headless", False)) and not bool(args_cli.disable_tiled_camera_inset):
            inset_args = argparse.Namespace(**vars(args_cli))
            inset_args.tiled_cam_prim_path = view_prim_path
            inset_window = _attach_tiled_camera_inset(inset_args)
        unwrapped._tiled_camera_inset_window = inset_window

        print(
            "[Info] Enabled eval tiled camera:"
            f" sensor_prim={sensor_prim_path},"
            f" inset_camera={view_prim_path},"
            f" inset={'off' if inset_window is None else 'on'}"
        )

    # =================================================================================
    # 定义 Bridge 类 (必须在导入 Node 之后)
    # =================================================================================
    class EnvRosBridge(NodeBase):
        """Runs the IsaacLab environment and exchanges data via ROS 2 topics."""

        def __init__(
            self,
            env,
            simulation_app,
            *,
            frame_id: str,
            depth_topic: str,
            odom_topic: str,
            cmd_topic: str,
            ctrl_attitude_des_topic: str,
            ctrl_attitude_real_topic: str,
            ctrl_bodyrate_des_topic: str,
            ctrl_bodyrate_real_topic: str,
            flatness_att_topic: str,
            flatness_rate_topic: str,
            flatness_thrust_topic: str,
            depth_shm_name: Optional[str],
            drone_id: int,
            mass: float,
            gravity: float,
        ) -> None:
            super().__init__("isaac_env_ros_bridge")
            self._ros_enabled = ros_enabled
            self._sim_app = simulation_app
            self._unwrapped = env.unwrapped
            self._device = self._unwrapped.device
            self._num_envs = self._unwrapped.num_envs
            self._drone_id = drone_id
            self._robot = self._resolve_robot()
            self._body_id = self._resolve_body_id()
            self._actions = torch.zeros((self._num_envs, 4), device=self._device)
            self._frame_id = frame_id
            self._mass = mass
            self._gravity = gravity
            self._physics_dt = float(self._unwrapped.physics_dt)
            self._env_step_dt = float(self._unwrapped.step_dt)
            self._latest_cmd = np.zeros(4, dtype=np.float32)
            self._last_depth: Optional[np.ndarray] = None
            self._last_state: Optional[np.ndarray] = None
            self._last_states: Optional[np.ndarray] = None
            self._last_att_quat = (1.0, 0.0, 0.0, 0.0)
            self._last_bodyrate = np.zeros(3, dtype=np.float64)
            self._current_sim_time = 0.0
            self._last_cmd_time = 0.0
            self._has_received_position_command = False
            self._urdf_px4_model = self._load_urdf_px4_model()
            self._kp = np.array(
                self._resolve_vector_param(args_cli.pos_kp, LEGACY_POS_KP, "pos_kp"),
                dtype=np.float32,
            )
            self._kd = np.array(
                self._resolve_vector_param(args_cli.vel_kp, LEGACY_VEL_KP, "vel_kp"),
                dtype=np.float32,
            )
            self._min_collective_acc = self._resolve_scalar_param(
                args_cli.min_collective_acc,
                LEGACY_MIN_COLLECTIVE_ACC,
                "min_collective_acc",
            )
            self._almost_zero_value_threshold = 1e-3
            self._flat_last_omg = np.zeros(3, dtype=np.float64)
            self._reset_pos: Optional[np.ndarray] = None
            self._reset_state: Optional[np.ndarray] = None
            self._reset_yaw: Optional[float] = None
            self._ang_kp = np.array(
                self._resolve_vector_param(
                    args_cli.attitude_fb_kp,
                    LEGACY_ATTITUDE_FB_KP,
                    "attitude_fb_kp",
                ),
                dtype=np.float64,
            )
            self._max_bodyrate_fb = self._resolve_scalar_param(
                args_cli.max_bodyrate_fb,
                LEGACY_MAX_BODYRATE_FB,
                "max_bodyrate_fb",
            )
            self._max_angle = math.radians(
                self._resolve_scalar_param(args_cli.max_angle_deg, LEGACY_MAX_ANGLE_DEG, "max_angle_deg")
            )
            self._last_flatness_debug = {
                "attitude": [0.0, 0.0, 0.0],
                "bodyrate": [0.0, 0.0, 0.0],
                "thrust_norm": 0.0,
            }
            self._depth_shm_name = depth_shm_name
            self._depth_shm_writer = DepthSharedMemoryWriter(depth_shm_name) if depth_shm_name else None
            self._use_ros2_sidecar = (not self._ros_enabled) and (not args_cli.disable_ros2_bridge)
            self._ros2_sidecar_process: Optional[subprocess.Popen] = None
            self._udp_cmd_socket: Optional[socket.socket] = None
            self._udp_state_socket: Optional[socket.socket] = None
            self._udp_state_target = (args_cli.ros2_state_host, int(args_cli.ros2_state_port))
            self._reset_log_path = _normalize_csv_log_path(args_cli.reset_log_path, default_filename="ego.csv")
            self._reset_log_target = max(0, int(args_cli.reset_log_count))
            self._reset_log_done = False
            self._reset_count = 0
            self._csv_header_written = False
            self._traj_buffers = [list() for _ in range(self._num_envs)]
            self._telemetry_log_path = _normalize_csv_log_path(
                args_cli.telemetry_log_path,
                default_filename=_default_hover_telemetry_filename(),
            )
            self._telemetry_log_file = None
            self._telemetry_writer = None
            self._last_reset_flags = np.zeros((self._num_envs,), dtype=bool)
            self._startup_hover_settle_steps = max(0, int(args_cli.startup_hover_settle_steps))
            self._enable_rotor_spin_visual = (not bool(getattr(args_cli, "headless", False))) and (not args_cli.disable_rotor_spin_visual)
            self._rotor_spin_visual_scale = max(0.0, float(args_cli.rotor_spin_visual_scale))
            self._rotor_joint_ids: list[int] = []
            self._rotor_joint_names: list[str] = []
            self._rotor_joint_angles: Optional[torch.Tensor] = None
            self._rotor_joint_zero_vel: Optional[torch.Tensor] = None
            self._rotor_joint_directions: Optional[torch.Tensor] = None
            self._tiled_camera_update_failed = False

            self._enable_rate_ctrl = False
            att_p_gain = self._resolve_vector_param(args_cli.att_p_gain, LEGACY_ATT_P_GAIN, "att_p_gain")
            rate_p_gain = self._resolve_vector_param(args_cli.rate_p_gain, LEGACY_RATE_P_GAIN, "rate_p_gain")
            rate_i_gain = self._resolve_vector_param(args_cli.rate_i_gain, LEGACY_RATE_I_GAIN, "rate_i_gain")
            rate_d_gain = self._resolve_vector_param(args_cli.rate_d_gain, LEGACY_RATE_D_GAIN, "rate_d_gain")
            rate_k_gain = self._resolve_vector_param(args_cli.rate_k_gain, LEGACY_RATE_K_GAIN, "rate_k_gain")
            rate_int_limit = self._resolve_vector_param(args_cli.rate_int_limit, LEGACY_RATE_INT_LIMIT, "rate_int_limit")
            att_rate_limit = self._resolve_vector_param(args_cli.att_rate_limit, LEGACY_ATT_RATE_LIMIT, "att_rate_limit")
            att_yaw_weight = self._resolve_scalar_param(args_cli.att_yaw_weight, LEGACY_ATT_YAW_WEIGHT, "att_yaw_weight")
            ctrl_model = self._resolve_controller_model()
            ctrl_inertia = torch.tensor(ctrl_model["inertia"], device=self._device, dtype=torch.float32)
            self._controller = PX4QuadrotorController(
                num_envs=self._num_envs,
                device=self._device,
                att_p_gain=torch.tensor(att_p_gain, device=self._device, dtype=torch.float32),
                att_yaw_weight=torch.tensor(att_yaw_weight, device=self._device, dtype=torch.float32),
                rate_p_gain=torch.tensor(rate_p_gain, device=self._device, dtype=torch.float32),
                rate_i_gain=torch.tensor(rate_i_gain, device=self._device, dtype=torch.float32),
                rate_d_gain=torch.tensor(rate_d_gain, device=self._device, dtype=torch.float32),
                rate_k_gain=torch.tensor(rate_k_gain, device=self._device, dtype=torch.float32),
                rate_int_limit=torch.tensor(rate_int_limit, device=self._device, dtype=torch.float32),
                att_rate_limit=torch.tensor(att_rate_limit, device=self._device, dtype=torch.float32),
                mass=torch.tensor(ctrl_model["mass"], device=self._device, dtype=torch.float32),
                inertia=ctrl_inertia,
                arm_length=torch.tensor(ctrl_model["arm_length"], device=self._device, dtype=torch.float32),
            )
            motor_omega_max = self._controller.dynamics.motor_omega_max_
            a_coef = ctrl_model["motor_thrust_max"] / torch.clamp(motor_omega_max * motor_omega_max, min=1e-6)
            self._controller.dynamics.thrust_map_[:, 0] = a_coef
            self._controller.dynamics.thrust_map_[:, 1] = 0.0
            self._controller.dynamics.thrust_map_[:, 2] = 0.0
            self._controller.dynamics.thrust_max_ = torch.full(
                (self._num_envs,), float(ctrl_model["motor_thrust_max"]), device=self._device, dtype=torch.float32
            )
            self._controller.dynamics.kappa_ = torch.full(
                (self._num_envs,), float(ctrl_model["kappa"]), device=self._device, dtype=torch.float32
            )
            self._controller.dynamics.set_thrust_ratio(
                self._resolve_scalar_param(
                    args_cli.ctrl_thrust_ratio,
                    LEGACY_CTRL_THRUST_RATIO,
                    "ctrl_thrust_ratio",
                )
            )
            ctrl_alloc_matrix = None
            urdf_allocation = self._urdf_px4_model.get("allocation_matrix")
            if urdf_allocation is not None:
                urdf_allocation = np.array(urdf_allocation, dtype=np.float32)
                if urdf_allocation.ndim == 2 and urdf_allocation.shape[0] >= 6:
                    ctrl_alloc_matrix = torch.tensor(
                        urdf_allocation[2:6, :],
                        device=self._device,
                        dtype=torch.float32,
                    ).unsqueeze(0).repeat(self._num_envs, 1, 1)
            if ctrl_alloc_matrix is None:
                ctrl_alloc_matrix = self._controller.dynamics.getAllocationMatrix()
            self._controller.alloc_matrix_ = ctrl_alloc_matrix
            self._controller.alloc_matrix_pinv_ = torch.linalg.pinv(self._controller.alloc_matrix_)
            self._controller.set_accel_filter_coef(
                self._resolve_scalar_param(args_cli.accel_filter_coef, LEGACY_ACCEL_FILTER_COEF, "accel_filter_coef")
            )
            self._ctrl_mass = float(self._controller.dynamics.mass_[0].item())
            self._ctrl_info = None

            if not hasattr(self._unwrapped, "_sim_step_counter"):
                self._unwrapped._sim_step_counter = 0

            self._cache_state()
            self._setup_rotor_spin_visual()
            self._refresh_hold_target(self._last_state, reason="startup", apply_command=True, log_update=True)
            self._run_startup_hover_settle()
            self._log_controller_model(ctrl_model)
            self._log_controller_tuning()

            if self._ros_enabled:
                self._depth_pub = self.create_publisher(Image, depth_topic, 10)
                self._odom_pub = self.create_publisher(Odometry, odom_topic, 10)
                self._reset_pub = self.create_publisher(Bool, "/drone_0_reset", 10)
                self._goal_pub = self.create_publisher(PoseStamped, "/move_base_simple/goal", 10)
                self._ctrl_att_des_pub = self.create_publisher(Vector3Stamped, ctrl_attitude_des_topic, 10)
                self._ctrl_att_real_pub = self.create_publisher(Vector3Stamped, ctrl_attitude_real_topic, 10)
                self._ctrl_bodyrate_des_pub = self.create_publisher(Vector3Stamped, ctrl_bodyrate_des_topic, 10)
                self._ctrl_bodyrate_real_pub = self.create_publisher(Vector3Stamped, ctrl_bodyrate_real_topic, 10)
                self._flatness_att_pub = self.create_publisher(Vector3Stamped, flatness_att_topic, 10)
                self._flatness_rate_pub = self.create_publisher(Vector3Stamped, flatness_rate_topic, 10)
                self._flatness_thrust_pub = self.create_publisher(Vector3Stamped, flatness_thrust_topic, 10)
                self._cmd_sub = self.create_subscription(PositionCommand, cmd_topic, self._on_position_command, 10)
            else:
                self._depth_pub = None
                self._odom_pub = None
                self._reset_pub = None
                self._goal_pub = None
                self._ctrl_att_des_pub = None
                self._ctrl_att_real_pub = None
                self._ctrl_bodyrate_des_pub = None
                self._ctrl_bodyrate_real_pub = None
                self._flatness_att_pub = None
                self._flatness_rate_pub = None
                self._flatness_thrust_pub = None
                self._cmd_sub = None

            if self._use_ros2_sidecar:
                self._setup_ros2_sidecar(depth_topic=depth_topic, odom_topic=odom_topic, cmd_topic=cmd_topic)
            elif not self._ros_enabled:
                self.get_logger().warning("ROS 2 topics disabled; using internal PX4 hover fallback only.")

        def _resolve_robot(self):
            robot = getattr(self._unwrapped, "_robot", None)
            if robot is not None:
                return robot
            scene = getattr(self._unwrapped, "scene", None)
            if scene is not None:
                with contextlib.suppress(Exception):
                    return scene["robot"]
            raise RuntimeError("Unable to locate robot asset on the environment.")

        def _resolve_body_id(self):
            for body_name in ("body", "base_link"):
                with contextlib.suppress(Exception):
                    return self._robot.find_bodies(body_name)[0]
            return 0

        def _setup_rotor_spin_visual(self) -> None:
            if not self._enable_rotor_spin_visual:
                return
            if self._rotor_spin_visual_scale <= 0.0:
                self.get_logger().info("Rotor spin visualization disabled: visual scale <= 0.")
                self._enable_rotor_spin_visual = False
                return
            urdf_model = self._urdf_px4_model or {}
            motors = urdf_model.get("motors") or []
            joint_names = [str(motor.get("joint_name", "")).strip() for motor in motors]
            if not joint_names or any(not name for name in joint_names):
                self.get_logger().warning("Rotor spin visualization disabled: motor joints not available from URDF model.")
                self._enable_rotor_spin_visual = False
                return
            try:
                joint_ids, resolved_joint_names = self._robot.find_joints(joint_names, preserve_order=True)
            except Exception as exc:
                self.get_logger().warning(f"Rotor spin visualization disabled: failed to resolve motor joints: {exc}")
                self._enable_rotor_spin_visual = False
                return
            if len(resolved_joint_names) != len(joint_names):
                self.get_logger().warning(
                    "Rotor spin visualization disabled:"
                    f" expected {len(joint_names)} joints, resolved {len(resolved_joint_names)}."
                )
                self._enable_rotor_spin_visual = False
                return

            self._rotor_joint_ids = [int(joint_id) for joint_id in joint_ids]
            self._rotor_joint_names = [str(name) for name in resolved_joint_names]
            self._rotor_joint_angles = torch.zeros(
                (self._num_envs, len(self._rotor_joint_ids)),
                device=self._device,
                dtype=torch.float32,
            )
            self._rotor_joint_zero_vel = torch.zeros_like(self._rotor_joint_angles)
            rotor_direction_by_name = {
                str(motor.get("joint_name", "")).strip(): float(motor.get("rotor_direction", 1.0))
                for motor in motors
            }
            missing_joint_names = [name for name in self._rotor_joint_names if name not in rotor_direction_by_name]
            if missing_joint_names:
                self.get_logger().warning(
                    "Rotor spin visualization disabled:"
                    f" missing rotor directions for joints: {', '.join(missing_joint_names)}."
                )
                self._enable_rotor_spin_visual = False
                return
            rotor_visual_compensation_by_name = {
                "motor1": 1.0,
                "motor2": -1.0,
                "motor3": -1.0,
                "motor4": 1.0,
            }
            rotor_directions = [
                rotor_direction_by_name[name] * rotor_visual_compensation_by_name.get(name, 1.0)
                for name in self._rotor_joint_names
            ]
            if len(rotor_directions) != len(self._rotor_joint_ids):
                self.get_logger().warning(
                    "Rotor spin visualization disabled:"
                    f" expected {len(self._rotor_joint_ids)} rotor directions, got {len(rotor_directions)}."
                )
                self._enable_rotor_spin_visual = False
                return
            # Some prop meshes are mirrored relative to others, so apply a visual-only
            # per-joint compensation to keep the apparent pairing as motor1/motor3
            # forward and motor2/motor4 reverse.
            self._rotor_joint_directions = torch.tensor(
                [rotor_directions],
                device=self._device,
                dtype=torch.float32,
            )
            self._write_rotor_spin_visual()
            self.get_logger().info(
                "Enabled rotor spin visualization on joints: "
                + ", ".join(self._rotor_joint_names)
                + f" (visual_scale={self._rotor_spin_visual_scale:.3f})"
            )

        def _write_rotor_spin_visual(self, env_ids: Sequence[int] | slice | None = None) -> None:
            if (
                not self._enable_rotor_spin_visual
                or self._rotor_joint_angles is None
                or self._rotor_joint_zero_vel is None
                or not self._rotor_joint_ids
            ):
                return
            if env_ids is None:
                position = self._rotor_joint_angles
                velocity = self._rotor_joint_zero_vel
            else:
                position = self._rotor_joint_angles[env_ids]
                velocity = self._rotor_joint_zero_vel[env_ids]
            self._robot.write_joint_state_to_sim(
                position=position,
                velocity=velocity,
                joint_ids=self._rotor_joint_ids,
                env_ids=env_ids,
            )

        def _update_rotor_spin_visual(
            self,
            motor_speeds: Optional[torch.Tensor],
            dt: float,
            *,
            force_sync: bool = False,
        ) -> None:
            if (
                not self._enable_rotor_spin_visual
                or motor_speeds is None
                or self._rotor_joint_angles is None
                or self._rotor_joint_directions is None
            ):
                return
            num_rotors = self._rotor_joint_angles.shape[1]
            signed_speed = (
                motor_speeds[:, :num_rotors].detach().to(dtype=torch.float32)
                * self._rotor_joint_directions
                * self._rotor_spin_visual_scale
            )
            self._rotor_joint_angles = torch.remainder(
                self._rotor_joint_angles + signed_speed * float(dt) + math.pi,
                2.0 * math.pi,
            ) - math.pi
            if force_sync:
                self._write_rotor_spin_visual()
                self._unwrapped.sim.forward()

        def _reset_rotor_spin_visual(self, env_ids: Sequence[int] | None = None) -> None:
            if not self._enable_rotor_spin_visual or self._rotor_joint_angles is None:
                return
            if env_ids is None:
                self._rotor_joint_angles.zero_()
                self._write_rotor_spin_visual()
                return
            self._rotor_joint_angles[env_ids] = 0.0
            self._write_rotor_spin_visual(env_ids=env_ids)

        def _body_index(self) -> int:
            body_id = self._body_id
            if isinstance(body_id, torch.Tensor):
                return int(body_id.reshape(-1)[0].item())
            if isinstance(body_id, np.ndarray):
                return int(body_id.reshape(-1)[0])
            if isinstance(body_id, (list, tuple)):
                return int(body_id[0])
            return int(body_id)

        def _load_urdf_px4_model(self) -> dict[str, object] | None:
            try:
                model = load_px4_robot_model(args_cli.ctrl_model_urdf, args_cli.ctrl_model_params)
            except Exception as exc:
                self.get_logger().warning(f"Failed to load URDF PX4 model defaults: {exc}")
                return None
            self.get_logger().info(
                "Loaded URDF PX4 model:"
                f" urdf={model.get('urdf_path')},"
                f" params={model.get('params_path') or '<none>'}"
            )
            return model

        def _resolve_scalar_param(self, cli_value: float, legacy_value: float, model_key: str) -> float:
            model = self._urdf_px4_model
            if model is None or model_key not in model:
                return float(cli_value)
            if math.isclose(float(cli_value), float(legacy_value), rel_tol=1e-6, abs_tol=1e-8):
                return float(model[model_key])
            return float(cli_value)

        def _resolve_vector_param(self, cli_value, legacy_value, model_key: str) -> list[float]:
            cli_list = [float(v) for v in cli_value]
            model = self._urdf_px4_model
            if model is None or model_key not in model:
                return cli_list
            legacy_list = [float(v) for v in legacy_value]
            if len(cli_list) != len(legacy_list):
                return cli_list
            if all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8) for a, b in zip(cli_list, legacy_list)):
                return [float(v) for v in model[model_key]]
            return cli_list

        def _resolve_controller_model(self) -> dict[str, object]:
            ctrl_mass = float(args_cli.ctrl_mass)
            ctrl_inertia = np.array(args_cli.ctrl_inertia, dtype=np.float32)
            ctrl_arm_length = float(args_cli.ctrl_arm_length)
            ctrl_motor_thrust_max = float(args_cli.ctrl_motor_thrust_max)
            ctrl_kappa = float(args_cli.ctrl_kappa)
            override_sources: list[str] = []
            urdf_model = self._urdf_px4_model or {}

            if math.isclose(ctrl_mass, LEGACY_CTRL_MASS, rel_tol=1e-6, abs_tol=1e-6) and "mass" in urdf_model:
                ctrl_mass = float(urdf_model["mass"])
                override_sources.append("mass<-urdf_model")

            if np.allclose(ctrl_inertia, LEGACY_CTRL_INERTIA_NP, rtol=1e-5, atol=1e-8) and "inertia" in urdf_model:
                ctrl_inertia = np.array(urdf_model["inertia"], dtype=np.float32)
                override_sources.append("inertia<-urdf_model")

            if math.isclose(ctrl_arm_length, LEGACY_CTRL_ARM_LENGTH, rel_tol=1e-6, abs_tol=1e-6) and "arm_length" in urdf_model:
                ctrl_arm_length = float(urdf_model["arm_length"])
                override_sources.append("arm_length<-urdf_model")

            if (
                math.isclose(ctrl_motor_thrust_max, LEGACY_CTRL_MOTOR_THRUST_MAX, rel_tol=1e-6, abs_tol=1e-6)
                and "motor_thrust_max" in urdf_model
            ):
                ctrl_motor_thrust_max = float(urdf_model["motor_thrust_max"])
                override_sources.append("motor_thrust_max<-urdf_model")

            if math.isclose(ctrl_kappa, LEGACY_CTRL_KAPPA, rel_tol=1e-6, abs_tol=1e-6) and "kappa" in urdf_model:
                ctrl_kappa = float(urdf_model["kappa"])
                override_sources.append("kappa<-urdf_model")

            robot_data = getattr(self._robot, "data", None)
            robot_cfg = getattr(self._robot, "cfg", None)

            default_mass = getattr(robot_data, "default_mass", None)
            if (
                math.isclose(ctrl_mass, LEGACY_CTRL_MASS, rel_tol=1e-6, abs_tol=1e-6)
                and default_mass is not None
                and "mass" not in urdf_model
            ):
                with contextlib.suppress(Exception):
                    ctrl_mass = float(default_mass[self._drone_id].sum().item())
                    override_sources.append("mass<-robot.default_mass")

            default_inertia = getattr(robot_data, "default_inertia", None)
            if (
                np.allclose(ctrl_inertia, LEGACY_CTRL_INERTIA_NP, rtol=1e-5, atol=1e-8)
                and default_inertia is not None
                and "inertia" not in urdf_model
            ):
                with contextlib.suppress(Exception):
                    inertia_raw = default_inertia[self._drone_id, self._body_index()].detach().cpu().numpy()
                    inertia_raw = np.array(inertia_raw, dtype=np.float32).reshape(-1)
                    if inertia_raw.size >= 9:
                        ctrl_inertia = np.array([inertia_raw[0], inertia_raw[4], inertia_raw[8]], dtype=np.float32)
                        override_sources.append("inertia<-robot.default_inertia")

            allocation_matrix = getattr(robot_cfg, "allocation_matrix", None)
            if (
                math.isclose(ctrl_arm_length, LEGACY_CTRL_ARM_LENGTH, rel_tol=1e-6, abs_tol=1e-6)
                and allocation_matrix is not None
                and "arm_length" not in urdf_model
            ):
                with contextlib.suppress(Exception):
                    alloc = np.array(allocation_matrix, dtype=np.float32)
                    if alloc.ndim == 2 and alloc.shape[0] >= 5:
                        inferred_arm = float(np.max(np.linalg.norm(alloc[3:5, :], axis=0)))
                        if inferred_arm > 1e-6:
                            ctrl_arm_length = inferred_arm
                            override_sources.append("arm_length<-robot.allocation_matrix")

            thruster_cfg = None
            actuators = getattr(robot_cfg, "actuators", None)
            if isinstance(actuators, dict):
                thruster_cfg = actuators.get("thrusters")

            if thruster_cfg is not None:
                thrust_range = getattr(thruster_cfg, "thrust_range", None)
                if (
                    math.isclose(ctrl_motor_thrust_max, LEGACY_CTRL_MOTOR_THRUST_MAX, rel_tol=1e-6, abs_tol=1e-6)
                    and thrust_range is not None
                    and len(thrust_range) >= 2
                    and "motor_thrust_max" not in urdf_model
                ):
                    ctrl_motor_thrust_max = float(thrust_range[1])
                    override_sources.append("motor_thrust_max<-thruster_cfg")

                torque_to_thrust_ratio = getattr(thruster_cfg, "torque_to_thrust_ratio", None)
                if (
                    math.isclose(ctrl_kappa, LEGACY_CTRL_KAPPA, rel_tol=1e-6, abs_tol=1e-6)
                    and torque_to_thrust_ratio is not None
                    and "kappa" not in urdf_model
                ):
                    ctrl_kappa = float(torque_to_thrust_ratio)
                    override_sources.append("kappa<-thruster_cfg")

            return {
                "mass": float(ctrl_mass),
                "inertia": ctrl_inertia.astype(np.float32),
                "arm_length": float(ctrl_arm_length),
                "motor_thrust_max": float(ctrl_motor_thrust_max),
                "kappa": float(ctrl_kappa),
                "sources": override_sources,
            }

        def _log_controller_model(self, ctrl_model: dict[str, object]) -> None:
            inertia_values = np.array(ctrl_model["inertia"], dtype=np.float32).tolist()
            source_text = ", ".join(ctrl_model.get("sources", [])) or "cli/defaults"
            self.get_logger().info(
                "Controller model: "
                f"mass={float(ctrl_model['mass']):.4f} kg, "
                f"inertia={inertia_values}, "
                f"arm_length={float(ctrl_model['arm_length']):.4f} m, "
                f"motor_thrust_max={float(ctrl_model['motor_thrust_max']):.4f} N, "
                f"kappa={float(ctrl_model['kappa']):.4f} "
                f"(source: {source_text})"
            )

        def _log_controller_tuning(self) -> None:
            self.get_logger().info(
                "Controller tuning: "
                f"pos_kp={self._kp.tolist()}, "
                f"vel_kp={self._kd.tolist()}, "
                f"attitude_fb_kp={self._ang_kp.tolist()}, "
                f"max_bodyrate_fb={float(self._max_bodyrate_fb):.4f}, "
                f"max_angle_deg={math.degrees(self._max_angle):.2f}, "
                f"min_collective_acc={float(self._min_collective_acc):.4f}, "
                f"thrust_ratio={float(self._controller.dynamics.thrust_ratio):.4f}, "
                f"accel_filter_coef={float(self._controller.accel_filter_coef):.4f}"
            )

        def _refresh_hold_target(
            self,
            state: Optional[np.ndarray],
            *,
            reason: str,
            apply_command: bool = False,
            log_update: bool = False,
        ) -> None:
            if state is None:
                return
            state_array = np.array(state, dtype=np.float32).reshape(-1)
            if state_array.size < 3:
                return
            self._reset_state = state_array.copy()
            self._reset_pos = state_array[:3].copy()
            if state_array.size >= 7:
                current_quat = tuple(float(v) for v in state_array[3:7])
                _, _, self._reset_yaw = self._quaternion_to_euler_zyx(current_quat)
            self._last_state = state_array.copy()
            if apply_command:
                self._set_hover_command()
            if log_update:
                yaw_text = f", yaw={self._reset_yaw:.3f}" if self._reset_yaw is not None else ""
                vel_text = ""
                if state_array.size >= 10:
                    vel = state_array[7:10]
                    vel_text = f", vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})"
                self.get_logger().info(
                    f"Updated hold target from {reason}: "
                    f"({self._reset_pos[0]:.3f}, {self._reset_pos[1]:.3f}, {self._reset_pos[2]:.3f})"
                    f"{yaw_text}{vel_text}"
                )

        def _apply_timeout_hold(self) -> None:
            # Keep the originally captured hold target when planner commands stop.
            # If the target is repeatedly refreshed to the current pose, any slow
            # descent becomes the new reference and the vehicle will settle to ground.
            if self._reset_pos is None:
                self._refresh_hold_target(self._last_state, reason="command_timeout_init", apply_command=True)
                return
            self._set_hover_command()

        def _update_reset_buffers(self) -> None:
            if hasattr(self._unwrapped, "_get_dones"):
                reset_terminated, reset_time_outs = self._unwrapped._get_dones()
            elif hasattr(self._unwrapped, "termination_manager"):
                self._unwrapped.reset_buf = self._unwrapped.termination_manager.compute()
                reset_terminated = self._unwrapped.termination_manager.terminated
                reset_time_outs = self._unwrapped.termination_manager.time_outs
            else:
                reset_terminated = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
                reset_time_outs = torch.zeros_like(reset_terminated)
            self._unwrapped.reset_terminated = reset_terminated
            self._unwrapped.reset_time_outs = reset_time_outs
            self._unwrapped.reset_buf = reset_terminated | reset_time_outs

        def _goal_position_for_publish(self) -> Optional[np.ndarray]:
            desired_pos = getattr(self._unwrapped, "_desired_pos_w", None)
            if desired_pos is not None:
                with contextlib.suppress(Exception):
                    return desired_pos[self._drone_id].detach().cpu().numpy()
            return None if self._reset_pos is None else np.array(self._reset_pos, copy=True)

        def _refresh_tiled_camera(self):
            if self._tiled_camera_update_failed:
                return None
            tiled_camera = getattr(self._unwrapped, "_tiled_camera", None)
            if tiled_camera is None:
                return None
            try:
                if (not getattr(tiled_camera, "_is_initialized", False)) or (not hasattr(tiled_camera, "_timestamp")):
                    tiled_camera._initialize_impl()
                    tiled_camera._is_initialized = True
                    with contextlib.suppress(Exception):
                        tiled_camera.reset()
                tiled_camera.update(self._physics_dt, force_recompute=True)
            except Exception as exc:
                self._tiled_camera_update_failed = True
                self.get_logger().warning(f"Tiled camera disabled after update failure: {exc}")
                self._unwrapped._tiled_camera = None
                return None
            return tiled_camera

        def _get_depth_tensor(self):
            tiled_camera = self._refresh_tiled_camera()
            if tiled_camera is None:
                return None
            depth_tensor = tiled_camera.data.output.get("depth", None)
            if depth_tensor is None:
                depth_tensor = tiled_camera.data.output.get("distance_to_image_plane", None)
            return depth_tensor

        def _setup_ros2_sidecar(self, *, depth_topic: str, odom_topic: str, cmd_topic: str) -> None:
            try:
                self._udp_cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._udp_cmd_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._udp_cmd_socket.bind((args_cli.ros2_cmd_host, int(args_cli.ros2_cmd_port)))
                self._udp_cmd_socket.setblocking(False)
                self._udp_state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._udp_state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError as exc:
                self._use_ros2_sidecar = False
                self.get_logger().warning(f"Failed to create ROS 2 sidecar UDP sockets: {exc}")
                return

            ros_setup = (
                f"source /opt/ros/{shlex.quote(args_cli.ros2_distro)}/setup.bash >/dev/null 2>&1; "
                f"[ -f {shlex.quote(args_cli.ros2_workspace)}/install/setup.bash ] && "
                f"source {shlex.quote(args_cli.ros2_workspace)}/install/setup.bash >/dev/null 2>&1 || true; "
            )
            bridge_args = [
                args_cli.ros2_bridge_python,
                "-m",
                "yopo_drone.utils.ros2_udp_bridge",
                "--frame-id",
                self._frame_id,
                "--depth-topic",
                depth_topic,
                "--odom-topic",
                odom_topic,
                "--cmd-topic",
                cmd_topic,
                "--cmd-host",
                args_cli.ros2_cmd_host,
                "--cmd-port",
                str(args_cli.ros2_cmd_port),
                "--state-host",
                args_cli.ros2_state_host,
                "--state-port",
                str(args_cli.ros2_state_port),
            ]
            if self._depth_shm_name:
                bridge_args.extend(["--depth-shm-name", self._depth_shm_name])
            bridge_cmd = ros_setup + " ".join(shlex.quote(arg) for arg in bridge_args)
            sidecar_env = os.environ.copy()
            for env_key in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
                sidecar_env.pop(env_key, None)
            try:
                self._ros2_sidecar_process = subprocess.Popen(
                    ["/bin/bash", "-lc", bridge_cmd],
                    cwd=str(_project_root()),
                    env=sidecar_env,
                )
                self.get_logger().info(
                    f"Started ROS 2 sidecar bridge on UDP {args_cli.ros2_state_host}:{args_cli.ros2_state_port}."
                )
            except OSError as exc:
                self._use_ros2_sidecar = False
                self.get_logger().warning(f"Failed to start ROS 2 sidecar bridge: {exc}")

        def _poll_sidecar_command(self) -> None:
            if not self._use_ros2_sidecar or self._udp_cmd_socket is None:
                return
            while True:
                try:
                    packet, _ = self._udp_cmd_socket.recvfrom(65535)
                except BlockingIOError:
                    break
                except OSError:
                    return
                try:
                    payload = json.loads(packet.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if payload.get("type") != "position_command":
                    continue
                pos = np.array(payload.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
                vel = np.array(payload.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
                acc = np.array(payload.get("acceleration", [0.0, 0.0, 0.0]), dtype=np.float32)
                yaw = float(payload.get("yaw", 0.0))
                yaw_dot = float(payload.get("yaw_dot", 0.0))
                self._apply_position_command(
                    pos,
                    vel,
                    acc,
                    np.zeros(3, dtype=np.float32),
                    yaw,
                    yaw_dot,
                )
                self._has_received_position_command = True
                self._last_cmd_time = self._current_sim_time

        def _send_sidecar_payload(self, payload: dict) -> None:
            if not self._use_ros2_sidecar or self._udp_state_socket is None:
                return
            try:
                self._udp_state_socket.sendto(json.dumps(payload).encode("utf-8"), self._udp_state_target)
            except OSError:
                return

        def run(self) -> None:
            render_interval = 1 / 30.0
            next_step = self._current_sim_time
            while self._sim_app.is_running() and (not self._ros_enabled or rclpy.ok()):
                if self._ros_enabled:
                    rclpy.spin_once(self, timeout_sec=0.0)
                if self._use_ros2_sidecar:
                    self._poll_sidecar_command()
                # Refresh the cached state before computing the fallback hold command.
                # This avoids using a stale startup pose on the first control step.
                self._cache_state()
                if (not self._has_received_position_command) or (self._current_sim_time - self._last_cmd_time > 1.0):
                    self._apply_timeout_hold()
                start_time = time.time()
                self._step_env()
                # NOTE: publishing image here will introduce significant delay
                # if self._current_sim_time > next_step:
                #     self._publish_depth(self._current_sim_time)
                #     next_step = sim_time + render_interval
                if (not self._has_received_position_command) or (self._current_sim_time - self._last_cmd_time > 1.0):
                    self._apply_timeout_hold()
                # print(f"Simulation time: {time.time() - start_time}")
                if self._reset_log_done:
                    break

        def _run_startup_hover_settle(self) -> None:
            if self._startup_hover_settle_steps <= 0 or self._reset_pos is None:
                return
            self.get_logger().info(
                f"Running startup hover settle for {self._startup_hover_settle_steps} steps before enabling telemetry."
            )
            for _ in range(self._startup_hover_settle_steps):
                self._cache_state()
                self._set_hover_command()
                self._run_controller_step(count_sim_step=False)
            self._cache_state()
            if self._last_state is not None:
                settle_state = np.array(self._last_state, dtype=np.float32)
                settle_pos = settle_state[:3]
                settle_vel = settle_state[7:10] if settle_state.size >= 10 else np.zeros(3, dtype=np.float32)
                settle_err = np.array(self._reset_pos, dtype=np.float32) - settle_pos
                settle_speed = float(np.linalg.norm(settle_vel))
                self.get_logger().info(
                    "Startup hover settle complete: "
                    f"state=({settle_pos[0]:.3f}, {settle_pos[1]:.3f}, {settle_pos[2]:.3f}), "
                    f"err_to_start=({settle_err[0]:.3f}, {settle_err[1]:.3f}, {settle_err[2]:.3f}), "
                    f"speed={settle_speed:.3f} m/s. "
                    "Keeping the original startup pose as the hover target."
                )
                self._set_hover_command()
            self._current_sim_time = 0.0
            self._last_cmd_time = 0.0
            self._unwrapped._sim_step_counter = 0

        def _step_env(self) -> None:
            self._run_controller_step(count_sim_step=True)

            self._unwrapped.episode_length_buf += 1
            self._unwrapped.common_step_counter += 1

            self._update_reset_buffers()
            self._last_reset_flags = self._unwrapped.reset_buf.detach().cpu().numpy().astype(bool)

            reset_env_ids = self._unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self._unwrapped._reset_idx(reset_env_ids)
                self._reset_rotor_spin_visual(reset_env_ids)
                self._unwrapped.scene.write_data_to_sim()
                self._unwrapped.sim.forward()
                self._unwrapped.sim.render()
                state = self._robot.data.root_state_w[self._drone_id].detach().cpu().numpy()
                self._refresh_hold_target(state, reason="env_reset", apply_command=True, log_update=True)
                self._controller.reset(reset_env_ids)

            if getattr(self._unwrapped, "_tiled_camera", None) is not None:
                self._unwrapped.sim.render()

            self._cache_state()
            sim_time = float(self._unwrapped._sim_step_counter) * self._physics_dt
            self._current_sim_time = sim_time
            self._cache_depth(sim_time)
            self._publish_odometry(sim_time)
            self._publish_ctrl_info(sim_time)
            self._publish_flatness(sim_time)
            self._log_telemetry(sim_time)
            reset_flags = self._last_reset_flags
            self._record_stats(sim_time, reset_flags)

            reset_now = bool(reset_flags[self._drone_id]) if reset_flags is not None and len(reset_flags) > self._drone_id else False
            if reset_now:
                if self._ros_enabled:
                    msg = Bool()
                    msg.data = True
                    self._reset_pub.publish(msg)
                    goal = PoseStamped()
                    self._fill_header(goal.header, sim_time)
                    pos = self._goal_position_for_publish()
                    if pos is not None:
                        goal.pose.position.x = float(pos[0])
                        goal.pose.position.y = float(pos[1])
                        goal.pose.position.z = float(pos[2])
                        goal.pose.orientation.w = 1.0
                        self._goal_pub.publish(goal)
                if self._use_ros2_sidecar:
                    self._send_sidecar_payload({"type": "reset", "stamp": sim_time, "data": True})
                    pos = self._goal_position_for_publish()
                    if pos is not None:
                        self._send_sidecar_payload(
                            {
                                "type": "goal",
                                "stamp": sim_time,
                                "frame_id": self._frame_id,
                                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                            }
                        )

        def _run_controller_step(self, *, count_sim_step: bool) -> None:
            cmd = torch.zeros(4, dtype=self._actions.dtype, device=self._device)
            cmd[: len(self._latest_cmd)] = torch.as_tensor(self._latest_cmd, device=self._device)[: 4]
            self._actions.zero_()
            self._actions[self._drone_id, : 4] = cmd

            with torch.inference_mode():
                actions = self._actions.to(self._device)

                for _ in range(self._unwrapped.cfg.decimation):
                    cur_state = self._robot.data.root_state_w.clone()
                    if self._enable_rate_ctrl:
                        _, _, motor_speeds, info = self._controller.compute_control(
                            cur_state, actions, self._physics_dt, mode='rate'
                        )
                    else:
                        _, _, motor_speeds, info = self._controller.compute_control(
                            cur_state, actions, self._physics_dt, mode='attitude'
                        )
                    self._ctrl_info = info

                    motor_thrusts = info.get("motor_thrusts")
                    if motor_thrusts is None:
                        raise RuntimeError("PX4 controller did not return motor thrust targets.")
                    self._robot.set_thrust_target(motor_thrusts)

                    if count_sim_step:
                        self._unwrapped._sim_step_counter += 1
                    self._unwrapped.scene.write_data_to_sim()
                    self._unwrapped.sim.step(render=False)
                    self._update_rotor_spin_visual(
                        motor_speeds,
                        self._physics_dt,
                        force_sync=(self._unwrapped._sim_step_counter % 4 == 0) if count_sim_step else False,
                    )
                    if count_sim_step and self._unwrapped._sim_step_counter % 4 == 0:
                        self._unwrapped.sim.render()
                    self._unwrapped.scene.update(dt=self._unwrapped.physics_dt)

        def _record_stats(self, timestamp: float, reset_flags: Optional[np.ndarray]) -> None:
            if self._reset_log_done or self._reset_log_target <= 0 or self._last_states is None:
                return
            if reset_flags is None:
                reset_flags = np.zeros((self._last_states.shape[0],), dtype=bool)
            esd_values = np.full((self._last_states.shape[0],), np.nan, dtype=np.float64)
            if all(hasattr(self._unwrapped, attr) for attr in ("grid_idx", "_maps")) and hasattr(self._unwrapped.cfg, "grid_rows") and hasattr(self._unwrapped.cfg, "grid_cols"):
                grid_rows = int(self._unwrapped.cfg.grid_rows)
                grid_cols = int(self._unwrapped.cfg.grid_cols)
                for i in range(grid_rows):
                    for j in range(grid_cols):
                        idx = i * grid_cols + j
                        env_ids = self._unwrapped.grid_idx[idx]
                        if not env_ids:
                            continue
                        env_ids = np.array(env_ids, dtype=int)
                        esd, _ = self._unwrapped._maps[idx].trilinear_interpolate_esdf(
                            self._robot.data.root_state_w[env_ids, :3].cpu().numpy()
                        )
                        esd_values[env_ids] = esd
            for env_id, state in enumerate(self._last_states):
                pos_x, pos_y, pos_z = float(state[0]), float(state[1]), float(state[2])
                vel_x, vel_y, vel_z = float(state[7]), float(state[8]), float(state[9])
                success = 1.0 if pos_x > 69.0 else 0.0
                reset_flag = 1.0 if reset_flags[env_id] else 0.0
                self._traj_buffers[env_id].append(
                    [timestamp, env_id, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, esd_values[env_id], success, reset_flag]
                )
                if reset_flags[env_id]:
                    self._write_reset_segment(self._traj_buffers[env_id])
                    self._traj_buffers[env_id].clear()
                    self._reset_count += 1
            if self._reset_count >= self._reset_log_target:
                self._reset_log_done = True
                print(f"Reached reset log target: {self._reset_count}")

        def _write_reset_segment(self, rows: list[list[float]]) -> None:
            if not rows:
                return
            write_header = not self._csv_header_written
            mode = "w" if write_header else "a"
            log_path = Path(self._reset_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open(mode, newline="") as csv_file:
                writer = csv.writer(csv_file)
                if write_header:
                    writer.writerow(
                        [
                            "timestamp",
                            "env_id",
                            "pos_x",
                            "pos_y",
                            "pos_z",
                            "vel_x",
                            "vel_y",
                            "vel_z",
                            "esd",
                            "success",
                            "reset",
                        ]
                    )
                    self._csv_header_written = True
                writer.writerows(rows)

        def _ensure_telemetry_writer(self) -> bool:
            if not self._telemetry_log_path:
                return False
            if self._telemetry_writer is not None and self._telemetry_log_file is not None:
                return True
            log_path = Path(self._telemetry_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._telemetry_log_file = log_path.open("w", newline="")
            self._telemetry_writer = csv.writer(self._telemetry_log_file)
            self._telemetry_writer.writerow(
                [
                    "timestamp",
                    "pos_x",
                    "pos_y",
                    "pos_z",
                    "vel_x",
                    "vel_y",
                    "vel_z",
                    "roll",
                    "pitch",
                    "yaw",
                    "bodyrate_x",
                    "bodyrate_y",
                    "bodyrate_z",
                    "target_x",
                    "target_y",
                    "target_z",
                    "target_yaw",
                    "err_x",
                    "err_y",
                    "err_z",
                    "speed_xy",
                    "speed_xyz",
                    "latest_cmd_0",
                    "latest_cmd_1",
                    "latest_cmd_2",
                    "latest_cmd_3",
                    "flat_roll",
                    "flat_pitch",
                    "flat_yaw",
                    "flat_bodyrate_x",
                    "flat_bodyrate_y",
                    "flat_bodyrate_z",
                    "flat_thrust_norm",
                    "des_roll",
                    "des_pitch",
                    "des_yaw",
                    "des_bodyrate_x",
                    "des_bodyrate_y",
                    "des_bodyrate_z",
                ]
            )
            self._telemetry_log_file.flush()
            print(f"[Info] Telemetry CSV logging enabled: {log_path}")
            return True

        def _log_telemetry(self, timestamp: float) -> None:
            if self._last_state is None or not self._ensure_telemetry_writer():
                return
            state = self._last_state
            pos = np.array(state[:3], dtype=np.float64)
            vel = np.array(state[7:10], dtype=np.float64)
            roll, pitch, yaw = self._quaternion_to_euler_zyx(tuple(float(v) for v in state[3:7]))
            bodyrate = np.array(state[10:13], dtype=np.float64)

            if self._reset_pos is None:
                target_pos = np.full(3, np.nan, dtype=np.float64)
            else:
                target_pos = np.array(self._reset_pos, dtype=np.float64)
            target_yaw = float(self._reset_yaw) if self._reset_yaw is not None else float("nan")
            pos_err = target_pos - pos if np.all(np.isfinite(target_pos)) else np.full(3, np.nan, dtype=np.float64)

            desired_euler = np.full(3, np.nan, dtype=np.float64)
            desired_bodyrate = np.full(3, np.nan, dtype=np.float64)
            ctrl = self._ctrl_info
            if ctrl is not None:
                q_des = ctrl.get("q_des")
                if q_des is not None:
                    if isinstance(q_des, torch.Tensor):
                        q_des = q_des.detach().cpu().numpy()
                    q_des = np.array(q_des)
                    if q_des.ndim >= 2 and q_des.shape[0] > self._drone_id:
                        desired_euler = np.array(
                            self._quaternion_to_euler_zyx(tuple(float(v) for v in q_des[self._drone_id].tolist())),
                            dtype=np.float64,
                        )
                rate_sp = ctrl.get("rate_sp")
                if rate_sp is not None:
                    if isinstance(rate_sp, torch.Tensor):
                        rate_sp = rate_sp.detach().cpu().numpy()
                    rate_sp = np.array(rate_sp)
                    if rate_sp.ndim >= 2 and rate_sp.shape[0] > self._drone_id:
                        desired_bodyrate = np.array(rate_sp[self._drone_id], dtype=np.float64)

            flat_att = np.array(self._last_flatness_debug["attitude"], dtype=np.float64)
            flat_bodyrate = np.array(self._last_flatness_debug["bodyrate"], dtype=np.float64)
            flat_thrust_norm = float(self._last_flatness_debug["thrust_norm"])
            latest_cmd = np.array(self._latest_cmd[:4], dtype=np.float64)

            self._telemetry_writer.writerow(
                [
                    float(timestamp),
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    float(vel[0]),
                    float(vel[1]),
                    float(vel[2]),
                    float(roll),
                    float(pitch),
                    float(yaw),
                    float(bodyrate[0]),
                    float(bodyrate[1]),
                    float(bodyrate[2]),
                    float(target_pos[0]),
                    float(target_pos[1]),
                    float(target_pos[2]),
                    float(target_yaw),
                    float(pos_err[0]),
                    float(pos_err[1]),
                    float(pos_err[2]),
                    float(np.linalg.norm(vel[:2])),
                    float(np.linalg.norm(vel)),
                    float(latest_cmd[0]),
                    float(latest_cmd[1]),
                    float(latest_cmd[2]),
                    float(latest_cmd[3]),
                    float(flat_att[0]),
                    float(flat_att[1]),
                    float(flat_att[2]),
                    float(flat_bodyrate[0]),
                    float(flat_bodyrate[1]),
                    float(flat_bodyrate[2]),
                    flat_thrust_norm,
                    float(desired_euler[0]),
                    float(desired_euler[1]),
                    float(desired_euler[2]),
                    float(desired_bodyrate[0]),
                    float(desired_bodyrate[1]),
                    float(desired_bodyrate[2]),
                ]
            )
            self._telemetry_log_file.flush()

        def _cache_state(self) -> None:
            states = self._robot.data.root_state_w.detach().cpu().numpy().copy()
            root_ang_vel_b = getattr(self._robot.data, "root_ang_vel_b", None)
            if root_ang_vel_b is not None and states.shape[1] >= 13:
                states[:, 10:13] = root_ang_vel_b.detach().cpu().numpy()
            self._last_states = states
            if 0 <= self._drone_id < states.shape[0]:
                self._last_state = states[self._drone_id]

        def _cache_depth(self, timestamp: float) -> None:
            depth_tensor = self._get_depth_tensor()
            if depth_tensor is None:
                return
            depth = depth_tensor[self._drone_id, :, :, 0].detach().cpu().numpy().astype(np.float32)
            self._last_depth = depth
            if self._depth_shm_writer is not None:
                try:
                    self._depth_shm_writer.write(depth, timestamp)
                except RuntimeError as exc:
                    self.get_logger().error(f"Depth shared memory error: {exc}")
                    self._depth_shm_writer.close()
                    self._depth_shm_writer = None

        def _publish_depth(self, timestamp: float) -> None:
            if not self._ros_enabled or self._last_depth is None:
                return
            depth_mm = (self._last_depth * 1000.0).clip(0, 65535).astype(np.uint16)
            msg = Image()
            self._fill_header(msg.header, timestamp)
            msg.height = depth_mm.shape[0]
            msg.width = depth_mm.shape[1]
            msg.encoding = "16UC1"
            msg.is_bigendian = 0
            msg.step = msg.width * 2
            msg.data = depth_mm.tobytes()
            self._depth_pub.publish(msg)

        def _publish_odometry(self, timestamp: float) -> None:
            if self._last_state is None:
                return
            state = self._last_state
            if self._use_ros2_sidecar:
                self._send_sidecar_payload(
                    {
                        "type": "odom",
                        "stamp": timestamp,
                        "frame_id": self._frame_id,
                        "child_frame_id": "drone_base",
                        "position": [float(state[0]), float(state[1]), float(state[2])],
                        "orientation": [float(state[3]), float(state[4]), float(state[5]), float(state[6])],
                        "linear_velocity": [float(state[7]), float(state[8]), float(state[9])],
                        "angular_velocity": [float(state[10]), float(state[11]), float(state[12])],
                    }
                )
            if not self._ros_enabled:
                return
            msg = Odometry()
            self._fill_header(msg.header, timestamp)
            msg.child_frame_id = "drone_base"
            msg.pose.pose.position.x = float(state[0])
            msg.pose.pose.position.y = float(state[1])
            msg.pose.pose.position.z = float(state[2])
            msg.pose.pose.orientation.w = float(state[3])
            msg.pose.pose.orientation.x = float(state[4])
            msg.pose.pose.orientation.y = float(state[5])
            msg.pose.pose.orientation.z = float(state[6])
            msg.twist.twist.linear.x = float(state[7])
            msg.twist.twist.linear.y = float(state[8])
            msg.twist.twist.linear.z = float(state[9])
            msg.twist.twist.angular.x = float(state[10])
            msg.twist.twist.angular.y = float(state[11])
            msg.twist.twist.angular.z = float(state[12])
            msg.pose.covariance = [0.0] * 36
            msg.twist.covariance = [0.0] * 36
            self._odom_pub.publish(msg)

        def _publish_ctrl_info(self, timestamp: float) -> None:
            ctrl = self._ctrl_info
            if ctrl is None or self._last_state is None:
                return

            def _to_numpy(value):
                if value is None:
                    return None
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                else:
                    value = np.array(value)
                return value

            if "roll_des" in ctrl and "pitch_des" in ctrl and "yaw_des" in ctrl:
                roll_des = _to_numpy(ctrl["roll_des"])[self._drone_id]
                pitch_des = _to_numpy(ctrl["pitch_des"])[self._drone_id]
                yaw_des = _to_numpy(ctrl["yaw_des"])[self._drone_id]
                desired_euler = np.array([roll_des, pitch_des, yaw_des], dtype=np.float64)
            elif "q_des" in ctrl:
                q_des = _to_numpy(ctrl["q_des"])[self._drone_id]
                desired_euler = np.array(self._quaternion_to_euler_zyx(tuple(q_des.tolist())), dtype=np.float64)
            else:
                desired_euler = np.zeros(3, dtype=np.float64)

            current_euler = np.array(self._quaternion_to_euler_zyx(tuple(self._last_state[3:7].tolist())), dtype=np.float64)
            rate_sp = _to_numpy(ctrl.get("rate_sp"))
            if rate_sp is not None:
                desired_bodyrates = np.array(rate_sp[self._drone_id], dtype=np.float64)
            else:
                desired_bodyrates = np.zeros(3, dtype=np.float64)
            current_bodyrates = np.array(self._last_state[10:13], dtype=np.float64)

            self._publish_vector(self._ctrl_att_des_pub, desired_euler, timestamp)
            self._publish_vector(self._ctrl_att_real_pub, current_euler, timestamp)
            self._publish_vector(self._ctrl_bodyrate_des_pub, desired_bodyrates, timestamp)
            self._publish_vector(self._ctrl_bodyrate_real_pub, current_bodyrates, timestamp)

        # [FIX] Added missing _publish_flatness method
        def _publish_flatness(self, timestamp: float) -> None:
            """Publish flatness controller debug info."""
            att = np.array(self._last_flatness_debug["attitude"], dtype=np.float64)
            rate = np.array(self._last_flatness_debug["bodyrate"], dtype=np.float64)
            # Pack scalar thrust into a vector
            thrust_vec = np.array([self._last_flatness_debug["thrust_norm"], 0.0, 0.0], dtype=np.float64)

            self._publish_vector(self._flatness_att_pub, att, timestamp)
            self._publish_vector(self._flatness_rate_pub, rate, timestamp)
            self._publish_vector(self._flatness_thrust_pub, thrust_vec, timestamp)

        def _publish_vector(self, publisher, values: np.ndarray, timestamp: float) -> None:
            if not self._ros_enabled or publisher is None:
                return
            msg = Vector3Stamped()
            self._fill_header(msg.header, timestamp)
            msg.header.frame_id = self._frame_id
            msg.vector.x = float(values[0])
            msg.vector.y = float(values[1])
            msg.vector.z = float(values[2])
            publisher.publish(msg)

        def _on_position_command(self, msg: PositionCommand) -> None:
            pos = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)
            vel = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z], dtype=np.float32)
            acc = np.array([msg.acceleration.x, msg.acceleration.y, msg.acceleration.z], dtype=np.float32)
            jerk = np.array([msg.jerk.x, msg.jerk.y, msg.jerk.z], dtype=np.float32) if hasattr(msg, "jerk") else np.zeros(3, dtype=np.float32)
            yaw = float(msg.yaw)
            yaw_dot = float(msg.yaw_dot)
            self._apply_position_command(pos, vel, acc, jerk, yaw, yaw_dot)
            self._has_received_position_command = True
            self._last_cmd_time = self._current_sim_time

        def _apply_position_command(
            self,
            pos_des: np.ndarray,
            vel_des: np.ndarray,
            acc_des: np.ndarray,
            jerk_des: np.ndarray,
            yaw_des: float,
            yaw_rate_des: float,
        ) -> None:
            if self._last_state is None:
                return
            pos = self._last_state[:3]
            vel = self._last_state[7:10]
            # pos_error = np.clip(pos_des - pos, -1.0, 1.0)
            # vel_error = np.clip(vel_des + self._kp * pos_error - vel, -1.0, 1.0)
            pos_error = pos_des - pos
            vel_error = vel_des + self._kp * pos_error - vel
            pid_acc = self._kd * vel_error
            total_acc = self._compute_limited_total_acc(pid_acc, acc_des)
            current_quat = np.array(self._last_state[3:7], dtype=np.float64)
            body_z = self._quaternion_to_matrix(current_quat) @ np.array([0, 0, 1])
            acc_along_z = float(np.dot(total_acc, body_z))
            acc_along_z = max(acc_along_z, self._min_collective_acc)
            jerk_vector = np.array(jerk_des, dtype=np.float64)
            attitude = self._flat_input_attitude(
                total_acc,
                jerk_vector,
                yaw_des,
                yaw_rate_des,
                np.array(current_quat, dtype=np.float64),
            )
            if attitude is None:
                quat = self._last_att_quat
                bodyrate = self._last_bodyrate
            else:
                quat, bodyrate = attitude
                self._last_att_quat = quat
                self._last_bodyrate = bodyrate
            fb_rates = self._compute_feedback_bodyrates(quat, current_quat)
            bodyrate = bodyrate + fb_rates
            self._last_bodyrate = bodyrate

            roll_raw, pitch_raw, yaw_raw = self._quaternion_to_euler_zyx(quat)
            max_roll = self._max_angle
            max_pitch = self._max_angle
            max_yaw = math.radians(180.0)

            scale = max(
                abs(roll_raw) / max_roll if max_roll > 0 else 0.0,
                abs(pitch_raw) / max_pitch if max_pitch > 0 else 0.0,
                abs(yaw_raw) / max_yaw if max_yaw > 0 else 0.0,
                1.0,
            )
            if scale > 1.0:
                roll = roll_raw / scale
                pitch = pitch_raw / scale
                yaw = yaw_raw / scale
            else:
                roll = roll_raw
                pitch = pitch_raw
                yaw = yaw_raw

            max_collective_force = (
                float(self._controller.dynamics.thrust_max_[0].item())
                * float(self._controller.dynamics.thrust_ratio)
                * 4.0
            )
            thrust_force = acc_along_z * self._ctrl_mass
            thrust_norm = np.clip(thrust_force / max(max_collective_force, 1e-6), 0.0, 1.0)

            self._latest_cmd.fill(0.0)
            if self._enable_rate_ctrl:
                self._latest_cmd[0] = self._last_bodyrate[0]
                self._latest_cmd[1] = self._last_bodyrate[1]
                self._latest_cmd[2] = self._last_bodyrate[2]
            else:
                self._latest_cmd[0] = roll
                self._latest_cmd[1] = pitch
                self._latest_cmd[2] = yaw

            self._latest_cmd[3] = thrust_norm

            self._last_flatness_debug = {
                "attitude": [float(roll), float(pitch), float(yaw)],
                "bodyrate": [float(bodyrate[0]), float(bodyrate[1]), float(bodyrate[2])],
                "thrust_norm": float(thrust_norm),
            }

        def _set_hover_command(self) -> None:
            if self._last_state is None or self._reset_pos is None:
                return
            yaw = float(self._reset_yaw) if self._reset_yaw is not None else 0.0
            self._apply_position_command(
                np.array(self._reset_pos, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                yaw,
                0.0,
            )

        def _fill_header(self, header, timestamp: float) -> None:
            sec = int(timestamp)
            nanosec = int((timestamp - sec) * 1e9)
            header.stamp.sec = sec
            header.stamp.nanosec = nanosec
            header.frame_id = self._frame_id

        def _compute_feedback_bodyrates(
            self, desired_quat: Tuple[float, float, float, float], current_quat: Tuple[float, float, float, float]
        ) -> np.ndarray:
            q_des = np.array(desired_quat, dtype=np.float64)
            q_est = np.array(current_quat, dtype=np.float64)
            q_err = self._quat_multiply(self._quat_conjugate(q_est), q_des)
            sign = 1.0 if q_err[0] >= 0.0 else -1.0
            fb = 2.0 * self._ang_kp * q_err[1:] * sign
            np.clip(fb, -self._max_bodyrate_fb, self._max_bodyrate_fb, out=fb)
            return fb

        def close(self) -> None:
            if self._telemetry_log_file is not None:
                self._telemetry_log_file.close()
                self._telemetry_log_file = None
                self._telemetry_writer = None
            inset_window = getattr(self._unwrapped, "_tiled_camera_inset_window", None)
            if inset_window is not None:
                with contextlib.suppress(Exception):
                    inset_window.destroy()
                self._unwrapped._tiled_camera_inset_window = None
            if hasattr(self._unwrapped, "_tiled_camera"):
                self._unwrapped._tiled_camera = None
            if hasattr(self._unwrapped, "_tiled_camera_view_prim_path"):
                self._unwrapped._tiled_camera_view_prim_path = None
            if self._depth_shm_writer is not None:
                self._depth_shm_writer.close()
            if self._udp_cmd_socket is not None:
                self._udp_cmd_socket.close()
                self._udp_cmd_socket = None
            if self._udp_state_socket is not None:
                self._udp_state_socket.close()
                self._udp_state_socket = None
            if self._ros2_sidecar_process is not None:
                self._ros2_sidecar_process.terminate()
                with contextlib.suppress(Exception):
                    self._ros2_sidecar_process.wait(timeout=5.0)
                self._ros2_sidecar_process = None
            self.destroy_node()

        def _compute_limited_total_acc(self, pid_error_acc: np.ndarray, ref_acc: np.ndarray) -> np.ndarray:
            total_acc = pid_error_acc + ref_acc
            total_acc[2] += self._gravity
            norm = np.linalg.norm(total_acc)
            if norm < self._min_collective_acc:
                total_acc = total_acc / norm * self._min_collective_acc
            z_acc = float(np.dot(total_acc, np.array([0.0, 0.0, 1.0], dtype=np.float64)))
            z_b = total_acc / np.linalg.norm(total_acc)
            if z_acc < self._min_collective_acc:
                z_acc = self._min_collective_acc
            rot_axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), z_b)
            norm_axis = np.linalg.norm(rot_axis)
            if norm_axis > 1e-9:
                rot_axis /= norm_axis
            rot_ang = math.acos(np.dot(np.array([0.0, 0.0, 1.0], dtype=np.float64), z_b))
            if rot_ang > self._max_angle:
                limited_z_b = self._axis_angle_rotation(rot_axis, self._max_angle, np.array([0.0, 0.0, 1.0], dtype=np.float64))
                total_acc = z_acc / math.cos(self._max_angle) * limited_z_b
            return total_acc

        @staticmethod
        def _quaternion_to_euler_zyx(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
            w, x, y, z = quat
            norm = math.sqrt(w * w + x * x + y * y + z * z)
            if norm <= 1e-9:
                return 0.0, 0.0, 0.0
            w /= norm
            x /= norm
            y /= norm
            z /= norm

            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (w * y - z * x)
            if abs(sinp) >= 1.0:
                pitch = math.copysign(math.pi / 2.0, sinp)
            else:
                pitch = math.asin(sinp)

            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return roll, pitch, yaw

        @staticmethod
        def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
            conj = quat.copy()
            conj[1:] = -conj[1:]
            return conj

        @staticmethod
        def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
                dtype=np.float64,
            )

        @staticmethod
        def _axis_angle_rotation(axis: np.ndarray, angle: float, vector: np.ndarray) -> np.ndarray:
            axis = axis / np.linalg.norm(axis)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            return (
                vector * cos_a
                + np.cross(axis, vector) * sin_a
                + axis * np.dot(axis, vector) * (1.0 - cos_a)
            )

        @staticmethod
        def _quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
            q = np.array(quat, dtype=np.float64)
            norm = np.linalg.norm(q)
            if norm < 1e-9:
                return np.eye(3)
            q = q / norm
            w, x, y, z = q
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z
            return np.array(
                [
                    [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                    [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                    [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
                ],
                dtype=np.float64,
            )

        def _normalize_with_grad(
            self, vec: np.ndarray, vec_dot: np.ndarray
        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            sqr_norm = float(np.dot(vec, vec))
            norm = math.sqrt(sqr_norm)
            unit = vec / norm
            grad = (vec_dot - vec * (np.dot(vec, vec_dot) / sqr_norm)) / norm
            return unit, grad

        def _flat_input_attitude(
            self,
            thr_acc: np.ndarray,
            jerk: np.ndarray,
            yaw: float,
            yaw_rate: float,
            att_est: np.ndarray,
        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            thr_norm = float(np.linalg.norm(thr_acc))
            att_est_array = np.array(att_est, dtype=np.float64)
            if thr_norm < self._min_collective_acc:
                return att_est_array, np.zeros(3, dtype=np.float64)

            normalized = self._normalize_with_grad(thr_acc, jerk)
            if normalized is None:
                return att_est_array, self._flat_last_omg.copy()
            zb, zbd = normalized

            syaw = math.sin(yaw)
            cyaw = math.cos(yaw)
            xc = np.array([cyaw, syaw, 0.0], dtype=np.float64)
            xcd = np.array([-syaw * yaw_rate, cyaw * yaw_rate, 0.0], dtype=np.float64)

            yc = np.cross(zb, xc)
            if np.linalg.norm(yc) < self._almost_zero_value_threshold:
                return att_est_array, self._flat_last_omg.copy()

            ycd = np.cross(zbd, xc) + np.cross(zb, xcd)
            normalized_y = self._normalize_with_grad(yc, ycd)
            yb, ybd = normalized_y

            xb = np.cross(yb, zb)
            xbd = np.cross(ybd, zb)

            omg = np.zeros(3, dtype=np.float64)
            omg[0] = 0.5 * (np.dot(zb, ybd) - np.dot(yb, zbd))
            omg[1] = 0.5 * (np.dot(xb, zbd) - np.dot(zb, xbd))
            omg[2] = 0.5 * (np.dot(yb, xbd) - np.dot(xb, ybd))

            rot_m = np.column_stack((xb, yb, zb))
            quat = self._rotation_matrix_to_quaternion(rot_m.tolist())
            quat_array = np.array(quat, dtype=np.float64)
            self._flat_last_omg = omg.copy()
            return quat_array, omg

        @staticmethod
        def _rotation_matrix_to_quaternion(rot: list[list[float]]) -> Tuple[float, float, float, float]:
            m00, m01, m02 = rot[0]
            m10, m11, m12 = rot[1]
            m20, m21, m22 = rot[2]
            trace = m00 + m11 + m22
            if trace > 0.0:
                s = math.sqrt(trace + 1.0) * 2.0
                qw = 0.25 * s
                qx = (m21 - m12) / s
                qy = (m02 - m20) / s
                qz = (m10 - m01) / s
            elif m00 > m11 and m00 > m22:
                s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
                qw = (m21 - m12) / s
                qx = 0.25 * s
                qy = (m01 + m10) / s
                qz = (m02 + m20) / s
            elif m11 > m22:
                s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
                qw = (m02 - m20) / s
                qx = (m01 + m10) / s
                qy = 0.25 * s
                qz = (m12 + m21) / s
            else:
                s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
                qw = (m10 - m01) / s
                qx = (m02 + m20) / s
                qy = (m12 + m21) / s
                qz = 0.25 * s
            return qw, qx, qy, qz

    # =================================================================================
    # Launch Logic
    # =================================================================================

    def _apply_editor_scene_deployment_tuning(env_cfg) -> None:
        if args_cli.disable_env_editor_scene_init:
            return

        adjustments: list[str] = []
        reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
        params = getattr(reset_base, "params", None)

        if isinstance(params, dict):
            pose_range = params.get("pose_range")
            if isinstance(pose_range, dict):
                z_range = pose_range.get("z")
                if z_range is not None and len(z_range) >= 2:
                    current_z = tuple(float(v) for v in z_range)
                    if not math.isclose(current_z[0], current_z[1], abs_tol=1e-8):
                        safe_z = (max(current_z[0], 0.75), max(current_z[1], 1.25))
                        if safe_z != current_z:
                            pose_range["z"] = safe_z
                            adjustments.append(f"reset_z={safe_z}")

                max_tilt = math.radians(10.0)
                for axis_name in ("roll", "pitch"):
                    angle_range = pose_range.get(axis_name)
                    if angle_range is None or len(angle_range) < 2:
                        continue
                    safe_angle = (max(float(angle_range[0]), -max_tilt), min(float(angle_range[1]), max_tilt))
                    if safe_angle != tuple(float(v) for v in angle_range):
                        pose_range[axis_name] = safe_angle
                        adjustments.append(f"{axis_name}={safe_angle}")

            velocity_range = params.get("velocity_range")
            if isinstance(velocity_range, dict):
                safe_speed = 0.05
                for axis_name in ("x", "y", "z", "roll", "pitch", "yaw"):
                    axis_range = velocity_range.get(axis_name)
                    if axis_range is None or len(axis_range) < 2:
                        continue
                    safe_range = (max(float(axis_range[0]), -safe_speed), min(float(axis_range[1]), safe_speed))
                    if safe_range != tuple(float(v) for v in axis_range):
                        velocity_range[axis_name] = safe_range
                        adjustments.append(f"vel_{axis_name}={safe_range}")

        episode_length = getattr(env_cfg, "episode_length_s", None)
        if episode_length is not None and float(episode_length) < 60.0:
            env_cfg.episode_length_s = 60.0
            adjustments.append("episode_length_s=60.0")

        if adjustments:
            print("[Info] Applied deployment reset tuning: " + ", ".join(adjustments))

    def _maybe_initialize_editor_scene() -> None:
        if args_cli.disable_env_editor_scene_init:
            return
        import isaaclab.sim as sim_utils
        try:
            from yopo_drone.env.drone_env_editor import initialize_scene_from_editor
        except ImportError:
            from e2e_drone.env.drone_env_editor import initialize_scene_from_editor

        initialize_scene_from_editor(
            sim_utils=sim_utils,
            world_path=args_cli.env_editor_world_path,
            create_new_stage=True,
            clear_existing_world=True,
            add_lights=not args_cli.disable_env_editor_lights,
            add_ground=not args_cli.disable_env_editor_ground,
        )
        sim_utils.update_stage()
        print(
            "[Info] Initialized stage from drone_env_editor helpers:"
            f" world_path={args_cli.env_editor_world_path},"
            f" ground={'off' if args_cli.disable_env_editor_ground else 'on'},"
            f" lights={'off' if args_cli.disable_env_editor_lights else 'on'}"
        )

    @hydra_task_config(args_cli.task, "")
    def _launch(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg) -> None:
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device:
            env_cfg.sim.device = args_cli.device
        if args_cli.sim_dt is not None:
            env_cfg.sim.dt = args_cli.sim_dt
        if args_cli.decimation is not None:
            env_cfg.decimation = args_cli.decimation
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        _apply_editor_scene_deployment_tuning(env_cfg)
        _maybe_initialize_editor_scene()
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env.reset()
        _maybe_create_eval_tiled_camera(env)

        bridge: Optional[EnvRosBridge] = None
        rclpy_inited = False

        if ros_enabled and not rclpy.ok():
            rclpy.init(args=None)
            rclpy_inited = True
        
        depth_shm_name = None
        if not args_cli.disable_depth_shm and args_cli.depth_shm_name:
            depth_shm_name = args_cli.depth_shm_name

        bridge = EnvRosBridge(
            env,
            simulation_app,
            frame_id=args_cli.frame_id,
            drone_id=args_cli.drone_id,
            depth_topic=args_cli.depth_topic,
            depth_shm_name=depth_shm_name,
            odom_topic=args_cli.odom_topic,
            cmd_topic=args_cli.cmd_topic,
            ctrl_attitude_des_topic=args_cli.ctrl_attitude_des_topic,
            ctrl_attitude_real_topic=args_cli.ctrl_attitude_real_topic,
            ctrl_bodyrate_des_topic=args_cli.ctrl_bodyrate_des_topic,
            ctrl_bodyrate_real_topic=args_cli.ctrl_bodyrate_real_topic,
            flatness_att_topic=args_cli.flatness_att_topic,
            flatness_rate_topic=args_cli.flatness_rate_topic,
            flatness_thrust_topic=args_cli.flatness_thrust_topic,
            mass=args_cli.mass,
            gravity=args_cli.gravity,
        )
        bridge.run()

        if bridge is not None:
            bridge.close()
        if ros_enabled and rclpy_inited:
            rclpy.shutdown()
        env.close()

    # 执行启动
    _launch()
    simulation_app.close()


if __name__ == "__main__":
    main()
