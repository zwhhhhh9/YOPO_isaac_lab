#!/usr/bin/env python3
"""
IsaacLab runner that streams depth/odometry via ROS 2.
Fix: Delays rclpy imports until AFTER Isaac Sim is initialized to avoid malloc corruption.
Fix: Added missing publishers and methods for flatness/control debug.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np
import torch

from isaaclab.app import AppLauncher
from e2e_drone.utils.px4_controller import PX4QuadrotorController


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
    parser.add_argument("--ctrl_mass", type=float, default=0.5, help="Controller model mass [kg].")
    parser.add_argument(
        "--ctrl_inertia",
        type=float,
        nargs=3,
        default=[4.8847e-4, 1.0395e-3, 1.21702e-3],
        help="Controller model inertia [Ixx Iyy Izz] in kg*m^2.",
    )
    parser.add_argument("--ctrl_arm_length", type=float, default=0.1827, help="Controller model motor arm length [m].")
    parser.add_argument("--ctrl_motor_thrust_max", type=float, default=4.5, help="Estimated per-motor max thrust [N].")
    parser.add_argument("--ctrl_thrust_ratio", type=float, default=0.6, help="Available thrust ratio for differential control.")
    parser.add_argument("--ctrl_kappa", type=float, default=0.015, help="Yaw moment ratio (kappa).")
    parser.add_argument("--reset_log_path", type=str, default="./e2e_drone/data/ego.csv", help="CSV path for reset stats.")
    parser.add_argument("--reset_log_count", type=int, default=26, help="Number of resets to log before stopping.")
    
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
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
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Vector3Stamped  # Added import
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool

    # 尝试导入自定义消息，如果没有 source 工作空间可能会失败
    try:
        from quadrotor_msgs.msg import PositionCommand
    except ImportError:
        print("[Error] Cannot import quadrotor_msgs. Make sure you sourced your ROS2 workspace.")
        sys.exit(1)

    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from e2e_drone import tasks
    import gymnasium as gym

    # =================================================================================
    # 定义 Bridge 类 (必须在导入 Node 之后)
    # =================================================================================
    class EnvRosBridge(Node):
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
            self._sim_app = simulation_app
            self._unwrapped = env.unwrapped
            self._device = self._unwrapped.device
            self._num_envs = self._unwrapped.num_envs
            self._drone_id = drone_id
            self._body_id = self._unwrapped._robot.find_bodies("body")[0]
            self._actions = torch.zeros((self._num_envs, 4), device=self._device)
            self._frame_id = frame_id
            self._mass = mass
            self._gravity = gravity
            self._step_dt = float(self._unwrapped.step_dt)
            self._latest_cmd = np.zeros(4, dtype=np.float32)
            self._last_depth: Optional[np.ndarray] = None
            self._last_state: Optional[np.ndarray] = None
            self._last_states: Optional[np.ndarray] = None
            self._last_att_quat = (1.0, 0.0, 0.0, 0.0)
            self._last_bodyrate = np.zeros(3, dtype=np.float64)
            self._current_sim_time = 0.0
            self._last_cmd_time = 0.0
            self._kp = np.array([2.0, 2.0, 2.0], dtype=np.float32)
            self._kd = np.array([8.0, 8.0, 8.0], dtype=np.float32)
            self._min_collective_acc = 3
            self._almost_zero_value_threshold = 1e-3
            self._flat_last_omg = np.zeros(3, dtype=np.float64)
            self._reset_pos: Optional[np.ndarray] = None
            self._reset_state: Optional[np.ndarray] = None
            self._ang_kp = np.array([20.0, 20.0, 20.0], dtype=np.float64)
            self._max_bodyrate_fb = 4.0
            self._max_angle = math.radians(30.0)
            self._last_flatness_debug = {
                "attitude": [0.0, 0.0, 0.0],
                "bodyrate": [0.0, 0.0, 0.0],
                "thrust_norm": 0.0,
            }
            self._depth_shm_writer = DepthSharedMemoryWriter(depth_shm_name) if depth_shm_name else None
            self._reset_log_path = args_cli.reset_log_path
            self._reset_log_target = max(0, int(args_cli.reset_log_count))
            self._reset_log_done = False
            self._reset_count = 0
            self._csv_header_written = False
            self._traj_buffers = [list() for _ in range(self._num_envs)]

            self._enable_rate_ctrl = False
            # Tuned for assets/robot./robot.urdf (0.5kg class quadrotor).
            att_p_gain = [4.5, 4.5, 2.0]
            rate_p_gain = [0.02, 0.03, 0.015]
            rate_i_gain = [0.02, 0.02, 0.015]
            rate_d_gain = [0.0015, 0.0020, 0.0008]
            rate_k_gain = [1.0, 1.0, 1.0]
            rate_int_limit = [0.2, 0.2, 0.15]
            att_rate_limit = [2.5, 2.5, 2.0]
            att_yaw_weight = 0.4
            ctrl_inertia = torch.tensor(args_cli.ctrl_inertia, device=self._device, dtype=torch.float32)
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
                mass=torch.tensor(args_cli.ctrl_mass, device=self._device, dtype=torch.float32),
                inertia=ctrl_inertia,
                arm_length=torch.tensor(args_cli.ctrl_arm_length, device=self._device, dtype=torch.float32),
            )
            # Motor model retune: T = a * omega^2, with per-motor thrust cap from args.
            motor_omega_max = self._controller.dynamics.motor_omega_max_
            a_coef = args_cli.ctrl_motor_thrust_max / torch.clamp(motor_omega_max * motor_omega_max, min=1e-6)
            self._controller.dynamics.thrust_map_[:, 0] = a_coef
            self._controller.dynamics.thrust_map_[:, 1] = 0.0
            self._controller.dynamics.thrust_map_[:, 2] = 0.0
            self._controller.dynamics.thrust_max_ = torch.full(
                (self._num_envs,), float(args_cli.ctrl_motor_thrust_max), device=self._device, dtype=torch.float32
            )
            self._controller.dynamics.kappa_ = torch.full(
                (self._num_envs,), float(args_cli.ctrl_kappa), device=self._device, dtype=torch.float32
            )
            self._controller.dynamics.set_thrust_ratio(float(args_cli.ctrl_thrust_ratio))
            self._controller.alloc_matrix_ = self._controller.dynamics.getAllocationMatrix()
            self._controller.alloc_matrix_pinv_ = torch.linalg.pinv(self._controller.alloc_matrix_)
            self._ctrl_mass = float(self._controller.dynamics.mass_[0].item())
            self._ctrl_info = None

            self._forces = torch.zeros(self._num_envs, 1, 3, device=self._device)
            self._torques = torch.zeros(self._num_envs, 1, 3, device=self._device)

            if not hasattr(self._unwrapped, "_sim_step_counter"):
                self._unwrapped._sim_step_counter = 0

            # Publishers
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

            # Subscriber
            self._cmd_sub = self.create_subscription(PositionCommand, cmd_topic, self._on_position_command, 10)

        def run(self) -> None:
            render_interval = 1 / 30.0
            next_step = self._current_sim_time
            while self._sim_app.is_running() and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)
                start_time = time.time()
                self._step_env()
                # NOTE: publishing image here will introduce significant delay
                # if self._current_sim_time > next_step:
                #     self._publish_depth(self._current_sim_time)
                #     next_step = sim_time + render_interval
                if self._current_sim_time- self._last_cmd_time > 1.0:
                    if self._unwrapped._robot.data.root_state_w.shape[self._drone_id] > 0:
                        state = self._unwrapped._robot.data.root_state_w[self._drone_id].detach().cpu().numpy()
                        self._reset_pos = state[:3]
                        self._set_hover_command()
                # print(f"Simulation time: {time.time() - start_time}")
                if self._reset_log_done:
                    break

        def _step_env(self) -> None:
            cmd = torch.zeros(4, dtype=self._actions.dtype, device=self._device)
            cmd[: len(self._latest_cmd)] = torch.as_tensor(self._latest_cmd, device=self._device)[: 4]
            self._actions.zero_()
            self._actions[self._drone_id, : 4] = cmd

            with torch.inference_mode():
                actions = self._actions.to(self._device)
                cur_state = self._unwrapped._robot.data.root_state_w.clone()
                if self._enable_rate_ctrl:
                    force, torque, _, info = self._controller.compute_control(cur_state, actions, self._step_dt, mode='rate')
                else:
                    force, torque, _, info = self._controller.compute_control(cur_state, actions, self._step_dt, mode='attitude')
                self._forces.zero_()
                self._torques.zero_()
                self._forces[:, 0, 2] = force[:, 2]
                self._torques[:, 0, :] = torque
                self._ctrl_info = info

                for i in range(self._unwrapped.cfg.decimation):
                    self._unwrapped._sim_step_counter += 1
                    self._unwrapped._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)
                    self._unwrapped.scene.write_data_to_sim()
                    self._unwrapped.sim.step(render=False)
                    if self._unwrapped._sim_step_counter % 4 == 0:
                        self._unwrapped.sim.render()
                    self._unwrapped.scene.update(dt=self._unwrapped.physics_dt)

                self._unwrapped.episode_length_buf += 1
                self._unwrapped.common_step_counter += 1

                self._unwrapped.reset_terminated[:], self._unwrapped.reset_time_outs[:] = self._unwrapped._get_dones()
                self._unwrapped.reset_buf = self._unwrapped.reset_terminated | self._unwrapped.reset_time_outs

                reset_env_ids = self._unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_env_ids) > 0:
                    self._unwrapped._reset_idx(reset_env_ids)
                    self._unwrapped.scene.write_data_to_sim()
                    self._unwrapped.sim.forward()
                    self._unwrapped.sim.render()
                    state = self._unwrapped._robot.data.root_state_w[self._drone_id].detach().cpu().numpy()
                    self._reset_state = state.copy()
                    self._controller.reset(reset_env_ids)

            self._cache_state()
            sim_time = float(self._unwrapped._sim_step_counter) * self._step_dt
            self._current_sim_time = sim_time
            self._cache_depth(sim_time)
            self._publish_odometry(sim_time)
            self._publish_ctrl_info(sim_time)
            self._publish_flatness(sim_time)
            reset_flags = None
            if self._unwrapped.reset_terminated.numel() > 0:
                reset_flags = self._unwrapped.reset_terminated.detach().cpu().numpy().astype(bool)
            self._record_stats(sim_time, reset_flags)

            reset_now = bool(self._unwrapped.reset_terminated[self._drone_id].item())
            if reset_now:
                msg = Bool()
                msg.data = True
                self._reset_pub.publish(msg)
                goal = PoseStamped()
                self._fill_header(goal.header, sim_time)
                pos = self._unwrapped._desired_pos_w[self._drone_id].detach().cpu().numpy()
                goal.pose.position.x = float(pos[0])
                goal.pose.position.y = float(pos[1])
                goal.pose.position.z = float(pos[2])
                goal.pose.orientation.w = 1.0
                self._goal_pub.publish(goal)

        def _record_stats(self, timestamp: float, reset_flags: Optional[np.ndarray]) -> None:
            if self._reset_log_done or self._reset_log_target <= 0 or self._last_states is None:
                return
            if reset_flags is None:
                reset_flags = np.zeros((self._last_states.shape[0],), dtype=bool)
            esd_values = np.full((self._last_states.shape[0],), np.nan, dtype=np.float64)
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
                        self._unwrapped._robot.data.root_state_w[env_ids, :3].cpu().numpy()
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
            with open(self._reset_log_path, mode, newline="") as csv_file:
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

        def _cache_state(self) -> None:
            states = self._unwrapped._robot.data.root_state_w.detach().cpu().numpy()
            states[:, 10:13] = self._unwrapped._robot.data.root_ang_vel_b.detach().cpu().numpy()
            self._last_states = states
            if 0 <= self._drone_id < states.shape[0]:
                self._last_state = states[self._drone_id]

        def _cache_depth(self, timestamp: float) -> None:
            depth_tensor = self._unwrapped._tiled_camera.data.output.get("depth", None)
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
            if self._last_depth is None:
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
            pos = self._last_state[:3]
            vel = self._last_state[7:10]
            pos_error = self._reset_pos - pos
            vel_error = self._kp * pos_error - vel
            translational_acc = self._kd * vel_error
            total_acc = np.array(
                [translational_acc[0], translational_acc[1], translational_acc[2] + self._gravity],
                dtype=np.float64,
            )
            max_collective_force = (
                float(self._controller.dynamics.thrust_max_[0].item())
                * float(self._controller.dynamics.thrust_ratio)
                * 4.0
            )
            hover_norm = total_acc[2] * self._ctrl_mass / max(max_collective_force, 1e-6)
            self._latest_cmd.fill(0.0)
            self._latest_cmd[3] = float(np.clip(hover_norm, 0.0, 1.0))

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
            if self._depth_shm_writer is not None:
                self._depth_shm_writer.close()
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

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
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

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env.reset()

        bridge: Optional[EnvRosBridge] = None
        rclpy_inited = False

        if not rclpy.ok():
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
        if rclpy_inited:
            rclpy.shutdown()
        env.close()

    # 执行启动
    _launch()
    simulation_app.close()


if __name__ == "__main__":
    main()
