#!/usr/bin/env python3
"""YOPO-style quintic trajectory generator adapted to the current controller.

This module migrates the quintic polynomial solver and yaw guidance logic from:
~/Documents/YOPO_local/YOPO_Origin/YOPO/YOPO/policy/poly_solver.py

It can be imported as a utility or run as a ROS 2 node that publishes
`quadrotor_msgs/PositionCommand` commands compatible with `eval_ego.py`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import signal
import socket
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from quadrotor_msgs.msg import PositionCommand
    from rclpy.node import Node
    from std_msgs.msg import Bool
except ImportError:
    rclpy = None
    Node = object
    Odometry = None
    PositionCommand = None
    Bool = None


DEFAULT_UDP_START_POS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


class Poly5Solver:
    """Scalar 5th-order polynomial solver migrated from YOPO."""

    def __init__(self, pos0: float, vel0: float, acc0: float, pos1: float, vel1: float, acc1: float, tf: float):
        if tf <= 0.0:
            raise ValueError("Quintic trajectory duration must be > 0.")
        state_mat = np.array([pos0, vel0, acc0, pos1, vel1, acc1], dtype=np.float64)
        t = float(tf)
        coef_inv = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1 / 2, 0, 0, 0],
                [-10 / t**3, -6 / t**2, -3 / (2 * t), 10 / t**3, -4 / t**2, 1 / (2 * t)],
                [15 / t**4, 8 / t**3, 3 / (2 * t**2), -15 / t**4, 7 / t**3, -1 / t**2],
                [-6 / t**5, -3 / t**4, -1 / (2 * t**3), 6 / t**5, -3 / t**4, 1 / (2 * t**3)],
            ],
            dtype=np.float64,
        )
        self.A = np.dot(coef_inv, state_mat)

    def get_snap(self, t: float | np.ndarray) -> float | np.ndarray:
        return 24 * self.A[4] + 120 * self.A[5] * t

    def get_jerk(self, t: float | np.ndarray) -> float | np.ndarray:
        return 6 * self.A[3] + 24 * self.A[4] * t + 60 * self.A[5] * t * t

    def get_acceleration(self, t: float | np.ndarray) -> float | np.ndarray:
        return 2 * self.A[2] + 6 * self.A[3] * t + 12 * self.A[4] * t * t + 20 * self.A[5] * t * t * t

    def get_velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        return (
            self.A[1]
            + 2 * self.A[2] * t
            + 3 * self.A[3] * t * t
            + 4 * self.A[4] * t * t * t
            + 5 * self.A[5] * t * t * t * t
        )

    def get_position(self, t: float | np.ndarray) -> float | np.ndarray:
        return (
            self.A[0]
            + self.A[1] * t
            + self.A[2] * t * t
            + self.A[3] * t * t * t
            + self.A[4] * t * t * t * t
            + self.A[5] * t * t * t * t * t
        )


class Polys5Solver:
    """Vectorized multi-trajectory 5th-order polynomial solver from YOPO."""

    def __init__(
        self,
        pos0: float,
        vel0: float,
        acc0: float,
        pos1: np.ndarray,
        vel1: np.ndarray,
        acc1: np.ndarray,
        tf: float,
    ):
        if tf <= 0.0:
            raise ValueError("Quintic trajectory duration must be > 0.")
        n = len(pos1)
        state_mat = np.array([[pos0] * n, [vel0] * n, [acc0] * n, pos1, vel1, acc1], dtype=np.float64)
        t = float(tf)
        coef_inv = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1 / 2, 0, 0, 0],
                [-10 / t**3, -6 / t**2, -3 / (2 * t), 10 / t**3, -4 / t**2, 1 / (2 * t)],
                [15 / t**4, 8 / t**3, 3 / (2 * t**2), -15 / t**4, 7 / t**3, -1 / t**2],
                [-6 / t**5, -3 / t**4, -1 / (2 * t**3), 6 / t**5, -3 / t**4, 1 / (2 * t**3)],
            ],
            dtype=np.float64,
        )
        self.A = np.dot(coef_inv, state_mat)

    def get_position(self, t: float | np.ndarray) -> np.ndarray:
        t = np.atleast_1d(t)
        result = (
            self.A[0][:, np.newaxis]
            + self.A[1][:, np.newaxis] * t
            + self.A[2][:, np.newaxis] * t**2
            + self.A[3][:, np.newaxis] * t**3
            + self.A[4][:, np.newaxis] * t**4
            + self.A[5][:, np.newaxis] * t**5
        )
        return result.flatten()


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_yaw(
    vel_dir: np.ndarray,
    goal_dir: np.ndarray,
    last_yaw: float,
    dt: float,
    max_yaw_rate: float = 0.5,
) -> tuple[float, float]:
    """YOPO yaw heuristic adapted for the current controller command format."""

    dt = max(float(dt), 1e-6)
    vel_dir = np.asarray(vel_dir, dtype=np.float64)
    goal_dir = np.asarray(goal_dir, dtype=np.float64)

    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)
    goal_dist = np.linalg.norm(goal_dir)
    goal_dir = goal_dir / (goal_dist + 1e-5)

    goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
    delta_yaw = wrap_to_pi(goal_yaw - last_yaw)
    weight = 6 * abs(delta_yaw) / np.pi

    dir_des = vel_dir + weight * goal_dir
    yaw_desired = np.arctan2(dir_des[1], dir_des[0]) if goal_dist > 0.5 else last_yaw

    yaw_diff = wrap_to_pi(yaw_desired - last_yaw)
    max_yaw_change = max_yaw_rate * np.pi * dt
    yaw_change = np.clip(yaw_diff, -max_yaw_change, max_yaw_change)

    yaw = wrap_to_pi(last_yaw + yaw_change)
    yawdot = yaw_change / dt
    return yaw, yawdot


@dataclass
class QuinticCommandSample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    yaw: float
    yaw_dot: float

    def as_eval_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        return (
            np.array(self.position, dtype=np.float32),
            np.array(self.velocity, dtype=np.float32),
            np.array(self.acceleration, dtype=np.float32),
            np.array(self.jerk, dtype=np.float32),
            float(self.yaw),
            float(self.yaw_dot),
        )

    def fill_position_command(self, command, *, trajectory_id: int, trajectory_flag: int) -> None:
        command.position.x = float(self.position[0])
        command.position.y = float(self.position[1])
        command.position.z = float(self.position[2])
        command.velocity.x = float(self.velocity[0])
        command.velocity.y = float(self.velocity[1])
        command.velocity.z = float(self.velocity[2])
        command.acceleration.x = float(self.acceleration[0])
        command.acceleration.y = float(self.acceleration[1])
        command.acceleration.z = float(self.acceleration[2])
        if hasattr(command, "jerk"):
            command.jerk.x = float(self.jerk[0])
            command.jerk.y = float(self.jerk[1])
            command.jerk.z = float(self.jerk[2])
        command.yaw = float(self.yaw)
        command.yaw_dot = float(self.yaw_dot)
        if hasattr(command, "kx"):
            command.kx = [0.0, 0.0, 0.0]
        if hasattr(command, "kv"):
            command.kv = [0.0, 0.0, 0.0]
        if hasattr(command, "trajectory_id"):
            command.trajectory_id = int(trajectory_id)
        if hasattr(command, "trajectory_flag"):
            command.trajectory_flag = int(trajectory_flag)

    def to_sidecar_payload(self) -> dict:
        return {
            "type": "position_command",
            "position": [float(v) for v in self.position],
            "velocity": [float(v) for v in self.velocity],
            "acceleration": [float(v) for v in self.acceleration],
            "yaw": float(self.yaw),
            "yaw_dot": float(self.yaw_dot),
        }


class QuinticTrajectory:
    """3D quintic trajectory that emits commands compatible with eval_ego."""

    def __init__(
        self,
        *,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        start_acc: np.ndarray,
        goal_pos: np.ndarray,
        goal_vel: np.ndarray,
        goal_acc: np.ndarray,
        duration: float,
        yaw_mode: str = "adaptive",
        initial_yaw: float = 0.0,
        fixed_yaw: float = 0.0,
        max_yaw_rate: float = 0.5,
    ):
        self.start_pos = np.asarray(start_pos, dtype=np.float64).reshape(3)
        self.start_vel = np.asarray(start_vel, dtype=np.float64).reshape(3)
        self.start_acc = np.asarray(start_acc, dtype=np.float64).reshape(3)
        self.goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
        self.goal_vel = np.asarray(goal_vel, dtype=np.float64).reshape(3)
        self.goal_acc = np.asarray(goal_acc, dtype=np.float64).reshape(3)
        self.duration = float(duration)
        self.yaw_mode = yaw_mode
        self.initial_yaw = float(initial_yaw)
        self.fixed_yaw = float(fixed_yaw)
        self.max_yaw_rate = float(max_yaw_rate)

        self._solvers = [
            Poly5Solver(
                self.start_pos[axis],
                self.start_vel[axis],
                self.start_acc[axis],
                self.goal_pos[axis],
                self.goal_vel[axis],
                self.goal_acc[axis],
                self.duration,
            )
            for axis in range(3)
        ]

    def sample(self, t: float, *, last_yaw: float, dt: float) -> QuinticCommandSample:
        t_eval = float(np.clip(t, 0.0, self.duration))
        position = np.array([solver.get_position(t_eval) for solver in self._solvers], dtype=np.float64)
        velocity = np.array([solver.get_velocity(t_eval) for solver in self._solvers], dtype=np.float64)
        acceleration = np.array([solver.get_acceleration(t_eval) for solver in self._solvers], dtype=np.float64)
        jerk = np.array([solver.get_jerk(t_eval) for solver in self._solvers], dtype=np.float64)

        yaw, yaw_dot = self._sample_yaw(position=position, velocity=velocity, last_yaw=last_yaw, dt=dt)
        return QuinticCommandSample(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            yaw=yaw,
            yaw_dot=yaw_dot,
        )

    def _sample_yaw(self, *, position: np.ndarray, velocity: np.ndarray, last_yaw: float, dt: float) -> tuple[float, float]:
        if self.yaw_mode == "adaptive":
            return calculate_yaw(
                vel_dir=velocity,
                goal_dir=self.goal_pos - position,
                last_yaw=last_yaw,
                dt=dt,
                max_yaw_rate=self.max_yaw_rate,
            )
        if self.yaw_mode == "initial":
            return self.initial_yaw, 0.0
        if self.yaw_mode == "fixed":
            return self.fixed_yaw, 0.0
        return 0.0, 0.0


def _quaternion_to_yaw(w: float, x: float, y: float, z: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class QuinticTrajectoryCommander(Node):
    """ROS 2 node that publishes YOPO-style quintic PositionCommand messages."""

    def __init__(
        self,
        *,
        odom_topic: str,
        cmd_topic: str,
        reset_topic: str,
        publish_rate_hz: float,
        frame_id: str,
        duration: float,
        goal_pos: np.ndarray,
        goal_vel: np.ndarray,
        goal_acc: np.ndarray,
        start_pos: Optional[np.ndarray],
        start_vel: Optional[np.ndarray],
        start_acc: np.ndarray,
        yaw_mode: str,
        fixed_yaw: float,
        max_yaw_rate: float,
    ) -> None:
        super().__init__("quintic_trajectory_commander")
        self._publisher = self.create_publisher(PositionCommand, cmd_topic, 10)
        self._odom_sub = self.create_subscription(Odometry, odom_topic, self._on_odometry, 10)
        self._reset_sub = self.create_subscription(Bool, reset_topic, self._on_reset, 10) if reset_topic else None
        self._timer = self.create_timer(1.0 / max(float(publish_rate_hz), 1e-3), self._on_timer)

        self._requested_frame_id = frame_id
        self._duration = float(duration)
        self._goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
        self._goal_vel = np.asarray(goal_vel, dtype=np.float64).reshape(3)
        self._goal_acc = np.asarray(goal_acc, dtype=np.float64).reshape(3)
        self._start_pos_override = None if start_pos is None else np.asarray(start_pos, dtype=np.float64).reshape(3)
        self._start_vel_override = None if start_vel is None else np.asarray(start_vel, dtype=np.float64).reshape(3)
        self._start_acc_override = np.asarray(start_acc, dtype=np.float64).reshape(3)
        self._yaw_mode = yaw_mode
        self._fixed_yaw = float(fixed_yaw)
        self._max_yaw_rate = float(max_yaw_rate)

        self._trajectory: Optional[QuinticTrajectory] = None
        self._trajectory_start_time = 0.0
        self._last_sample_time = 0.0
        self._last_yaw = 0.0
        self._trajectory_id = 1
        self._target_frame_id = frame_id or "world"
        self._waiting_log_count = 0
        self._done_log_sent = False

        self.get_logger().info(
            f"Waiting for odometry on '{odom_topic}' and publishing quintic PositionCommand to '{cmd_topic}'."
        )

    def _on_odometry(self, msg: Odometry) -> None:
        if self._trajectory is not None:
            return

        start_pos = (
            self._start_pos_override
            if self._start_pos_override is not None
            else np.array(
                [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
                dtype=np.float64,
            )
        )
        start_vel = (
            self._start_vel_override
            if self._start_vel_override is not None
            else np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
                dtype=np.float64,
            )
        )
        start_acc = self._start_acc_override
        captured_yaw = _quaternion_to_yaw(
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        )

        if self._yaw_mode == "initial":
            initial_yaw = captured_yaw
        elif self._yaw_mode == "fixed":
            initial_yaw = self._fixed_yaw
        else:
            initial_yaw = 0.0 if self._yaw_mode == "zero" else captured_yaw

        self._target_frame_id = self._requested_frame_id or msg.header.frame_id or "world"
        self._last_yaw = initial_yaw
        self._trajectory = QuinticTrajectory(
            start_pos=start_pos,
            start_vel=start_vel,
            start_acc=start_acc,
            goal_pos=self._goal_pos,
            goal_vel=self._goal_vel,
            goal_acc=self._goal_acc,
            duration=self._duration,
            yaw_mode=self._yaw_mode,
            initial_yaw=initial_yaw,
            fixed_yaw=self._fixed_yaw,
            max_yaw_rate=self._max_yaw_rate,
        )
        now = float(self.get_clock().now().nanoseconds) / 1e9
        self._trajectory_start_time = now
        self._last_sample_time = now
        self._done_log_sent = False

        self.get_logger().info(
            "Started quintic trajectory: "
            f"start=({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}), "
            f"goal=({self._goal_pos[0]:.3f}, {self._goal_pos[1]:.3f}, {self._goal_pos[2]:.3f}), "
            f"goal_vel=({self._goal_vel[0]:.3f}, {self._goal_vel[1]:.3f}, {self._goal_vel[2]:.3f}), "
            f"goal_acc=({self._goal_acc[0]:.3f}, {self._goal_acc[1]:.3f}, {self._goal_acc[2]:.3f}), "
            f"duration={self._duration:.3f}s, yaw_mode={self._yaw_mode}."
        )

    def _on_reset(self, msg: Bool) -> None:
        if not msg.data:
            return
        self._trajectory = None
        self._trajectory_id += 1
        self._done_log_sent = False
        self.get_logger().info("Reset received. Waiting for a fresh odometry sample to replan the quintic trajectory.")

    def _on_timer(self) -> None:
        if self._trajectory is None:
            self._waiting_log_count += 1
            if self._waiting_log_count % 100 == 0:
                self.get_logger().info("Still waiting for an odometry sample to initialize the quintic trajectory...")
            return

        now = float(self.get_clock().now().nanoseconds) / 1e9
        elapsed = max(0.0, now - self._trajectory_start_time)
        dt = max(now - self._last_sample_time, 1e-3)
        sample = self._trajectory.sample(elapsed, last_yaw=self._last_yaw, dt=dt)

        command = PositionCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self._target_frame_id
        sample.fill_position_command(
            command,
            trajectory_id=self._trajectory_id,
            trajectory_flag=getattr(PositionCommand, "TRAJECTORY_STATUS_READY", 1),
        )
        self._publisher.publish(command)

        self._last_yaw = sample.yaw
        self._last_sample_time = now

        if elapsed >= self._duration and not self._done_log_sent:
            self._done_log_sent = True
            self.get_logger().info(
                "Quintic trajectory reached terminal sample and will keep publishing the terminal command "
                "for controller hold."
            )


class UdpQuinticTrajectoryCommander:
    """Publish quintic commands straight to eval_ego's UDP sidecar input."""

    def __init__(
        self,
        *,
        udp_host: str,
        udp_port: int,
        publish_rate_hz: float,
        frame_id: str,
        duration: float,
        goal_pos: np.ndarray,
        goal_vel: np.ndarray,
        goal_acc: np.ndarray,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        start_acc: np.ndarray,
        yaw_mode: str,
        start_yaw: float,
        fixed_yaw: float,
        max_yaw_rate: float,
    ) -> None:
        self._target = (str(udp_host), int(udp_port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._publish_period = 1.0 / max(float(publish_rate_hz), 1e-3)
        self._frame_id = frame_id or "world"
        self._duration = float(duration)
        self._start_time = time.monotonic()
        self._last_sample_time = self._start_time
        self._last_yaw = float(fixed_yaw) if yaw_mode == "fixed" else float(start_yaw)
        self._done_log_sent = False
        self._trajectory = QuinticTrajectory(
            start_pos=np.asarray(start_pos, dtype=np.float64).reshape(3),
            start_vel=np.asarray(start_vel, dtype=np.float64).reshape(3),
            start_acc=np.asarray(start_acc, dtype=np.float64).reshape(3),
            goal_pos=np.asarray(goal_pos, dtype=np.float64).reshape(3),
            goal_vel=np.asarray(goal_vel, dtype=np.float64).reshape(3),
            goal_acc=np.asarray(goal_acc, dtype=np.float64).reshape(3),
            duration=self._duration,
            yaw_mode=yaw_mode,
            initial_yaw=float(start_yaw),
            fixed_yaw=float(fixed_yaw),
            max_yaw_rate=float(max_yaw_rate),
        )

    def run(self) -> None:
        print(
            "Publishing quintic UDP commands to "
            f"{self._target[0]}:{self._target[1]} "
            f"(duration={self._duration:.3f}s, frame_id='{self._frame_id}')."
        )
        while True:
            loop_start = time.monotonic()
            elapsed = max(0.0, loop_start - self._start_time)
            dt = max(loop_start - self._last_sample_time, 1e-3)
            sample = self._trajectory.sample(elapsed, last_yaw=self._last_yaw, dt=dt)
            payload = sample.to_sidecar_payload()
            payload["stamp"] = time.time()
            payload["frame_id"] = self._frame_id
            self._socket.sendto(json.dumps(payload).encode("utf-8"), self._target)
            self._last_yaw = sample.yaw
            self._last_sample_time = loop_start

            if elapsed >= self._duration and not self._done_log_sent:
                self._done_log_sent = True
                print("Quintic UDP trajectory reached the terminal sample and will keep publishing hold commands.")

            sleep_time = self._publish_period - (time.monotonic() - loop_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    def close(self) -> None:
        self._socket.close()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a YOPO-style quintic trajectory and publish PositionCommand to the current controller.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--transport",
        choices=("ros2", "udp"),
        default="udp",
        help="Command transport. 'udp' is the default because it matches eval_ego's working sidecar path.",
    )
    parser.add_argument("--odom-topic", type=str, default="/drone_0_odometry", help="Odometry topic published by eval_ego.")
    parser.add_argument(
        "--cmd-topic",
        type=str,
        default="/drone_0_planning/pos_cmd",
        help="PositionCommand topic consumed by eval_ego.",
    )
    parser.add_argument(
        "--reset-topic",
        type=str,
        default="/drone_0_reset",
        help="Reset topic used to replan the quintic trajectory after env reset. Use empty string to disable.",
    )
    parser.add_argument("--publish-rate", type=float, default=50.0, help="Trajectory command publish rate in Hz.")
    parser.add_argument("--frame-id", type=str, default="", help="Frame id written into PositionCommand header.")
    parser.add_argument("--udp-host", type=str, default="127.0.0.1", help="UDP host used when --transport=udp.")
    parser.add_argument("--udp-port", type=int, default=15000, help="UDP port used when --transport=udp.")
    parser.add_argument("--duration", type=float, default=5.0, help="Quintic trajectory duration in seconds.")
    parser.add_argument(
        "--goal-pos",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Goal world-frame position.",
    )
    parser.add_argument(
        "--goal-vel",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "VZ"),
        help="Goal world-frame velocity.",
    )
    parser.add_argument(
        "--goal-acc",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("AX", "AY", "AZ"),
        help="Goal world-frame acceleration.",
    )
    parser.add_argument(
        "--start-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional start position override. Default uses the first odometry sample.",
    )
    parser.add_argument(
        "--start-vel",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "VZ"),
        help="Optional start velocity override. Default uses the first odometry sample.",
    )
    parser.add_argument(
        "--start-acc",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("AX", "AY", "AZ"),
        help="Start acceleration used by the quintic boundary conditions.",
    )
    parser.add_argument(
        "--start-yaw",
        type=float,
        default=0.0,
        help="Start yaw used by --transport=udp and as the fallback initial yaw.",
    )
    parser.add_argument(
        "--yaw-mode",
        choices=("adaptive", "initial", "zero", "fixed"),
        default="adaptive",
        help="Yaw generation mode. 'adaptive' uses the original YOPO yaw heuristic.",
    )
    parser.add_argument("--fixed-yaw", type=float, default=0.0, help="Yaw used when --yaw-mode=fixed.")
    parser.add_argument("--max-yaw-rate", type=float, default=0.5, help="Max yaw rate factor used by adaptive yaw.")
    return parser


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    if args.duration <= 0.0:
        parser.error("--duration must be > 0.")
    if args.publish_rate <= 0.0:
        parser.error("--publish-rate must be > 0.")
    if args.transport == "udp":
        if args.start_pos is None:
            print(
                "No --start-pos provided for UDP transport. "
                f"Using the default hover start position {DEFAULT_UDP_START_POS.tolist()}."
            )
            start_pos = DEFAULT_UDP_START_POS.copy()
        else:
            start_pos = np.array(args.start_pos, dtype=np.float64)
        start_vel = np.zeros(3, dtype=np.float64) if args.start_vel is None else np.array(args.start_vel, dtype=np.float64)
        commander = UdpQuinticTrajectoryCommander(
            udp_host=args.udp_host,
            udp_port=args.udp_port,
            publish_rate_hz=args.publish_rate,
            frame_id=args.frame_id,
            duration=args.duration,
            goal_pos=np.array(args.goal_pos, dtype=np.float64),
            goal_vel=np.array(args.goal_vel, dtype=np.float64),
            goal_acc=np.array(args.goal_acc, dtype=np.float64),
            start_pos=start_pos,
            start_vel=start_vel,
            start_acc=np.array(args.start_acc, dtype=np.float64),
            yaw_mode=args.yaw_mode,
            start_yaw=args.start_yaw,
            fixed_yaw=args.fixed_yaw,
            max_yaw_rate=args.max_yaw_rate,
        )

        def _handle_udp_signal(_signum, _frame) -> None:
            raise KeyboardInterrupt

        previous_sigint = signal.signal(signal.SIGINT, _handle_udp_signal)
        previous_sigterm = signal.signal(signal.SIGTERM, _handle_udp_signal)
        try:
            commander.run()
        except KeyboardInterrupt:
            pass
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
            commander.close()
        return 0

    if rclpy is None or PositionCommand is None or Odometry is None:
        print("This script requires a sourced ROS 2 environment with nav_msgs and quadrotor_msgs available.")
        return 1

    rclpy.init(args=None)
    node = QuinticTrajectoryCommander(
        odom_topic=args.odom_topic,
        cmd_topic=args.cmd_topic,
        reset_topic=args.reset_topic,
        publish_rate_hz=args.publish_rate,
        frame_id=args.frame_id,
        duration=args.duration,
        goal_pos=np.array(args.goal_pos, dtype=np.float64),
        goal_vel=np.array(args.goal_vel, dtype=np.float64),
        goal_acc=np.array(args.goal_acc, dtype=np.float64),
        start_pos=None if args.start_pos is None else np.array(args.start_pos, dtype=np.float64),
        start_vel=None if args.start_vel is None else np.array(args.start_vel, dtype=np.float64),
        start_acc=np.array(args.start_acc, dtype=np.float64),
        yaw_mode=args.yaw_mode,
        fixed_yaw=args.fixed_yaw,
        max_yaw_rate=args.max_yaw_rate,
    )

    def _handle_signal(_signum, _frame) -> None:
        if rclpy.ok():
            with contextlib.suppress(Exception):
                rclpy.shutdown()

    previous_sigint = signal.signal(signal.SIGINT, _handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if "context is not valid" not in str(exc) and "ExternalShutdownException" not in type(exc).__name__:
            raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        with contextlib.suppress(Exception):
            node.destroy_node()
        if rclpy.ok():
            with contextlib.suppress(Exception):
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
