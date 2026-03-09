#!/usr/bin/env python3
"""Publish a hover PositionCommand using the drone's captured initial pose.

Start this node after `eval_ego.py` is running. The node waits for the first
odometry message, stores that pose as the hover target, and then keeps
publishing a `PositionCommand` so the PX4-style controller holds the drone at
that position.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import math
from typing import Optional

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


def _quaternion_to_yaw(w: float, x: float, y: float, z: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class HoverInitialPositionCommander(Node):
    def __init__(
        self,
        *,
        odom_topic: str,
        cmd_topic: str,
        reset_topic: str,
        publish_rate_hz: float,
        frame_id: str,
        yaw_mode: str,
        x_offset: float,
        y_offset: float,
        z_offset: float,
    ) -> None:
        super().__init__("hover_initial_position_commander")
        self._publisher = self.create_publisher(PositionCommand, cmd_topic, 10)
        self._odom_sub = self.create_subscription(Odometry, odom_topic, self._on_odometry, 10)
        self._reset_sub = self.create_subscription(Bool, reset_topic, self._on_reset, 10) if reset_topic else None
        self._timer = self.create_timer(1.0 / max(publish_rate_hz, 1e-3), self._on_timer)

        self._requested_frame_id = frame_id
        self._yaw_mode = yaw_mode
        self._offset = (float(x_offset), float(y_offset), float(z_offset))

        self._target_position: Optional[tuple[float, float, float]] = None
        self._target_yaw = 0.0
        self._target_frame_id = frame_id or "world"
        self._capture_next_odom = True
        self._trajectory_id = 1
        self._waiting_log_count = 0

        self.get_logger().info(
            f"Waiting for odometry on '{odom_topic}' and publishing hover commands to '{cmd_topic}'."
        )

    def _on_odometry(self, msg: Odometry) -> None:
        if not self._capture_next_odom:
            return

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        captured_yaw = _quaternion_to_yaw(orientation.w, orientation.x, orientation.y, orientation.z)

        self._target_position = (
            float(position.x) + self._offset[0],
            float(position.y) + self._offset[1],
            float(position.z) + self._offset[2],
        )
        self._target_yaw = captured_yaw if self._yaw_mode == "initial" else 0.0
        self._target_frame_id = self._requested_frame_id or msg.header.frame_id or "world"
        self._capture_next_odom = False

        self.get_logger().info(
            "Captured hover target: "
            f"position=({self._target_position[0]:.3f}, {self._target_position[1]:.3f}, {self._target_position[2]:.3f}), "
            f"yaw={self._target_yaw:.3f} rad, frame_id='{self._target_frame_id}'."
        )

    def _on_reset(self, msg: Bool) -> None:
        if not msg.data:
            return
        self._capture_next_odom = True
        self._target_position = None
        self._trajectory_id += 1
        self.get_logger().info("Reset received. Waiting for a fresh odometry sample to recapture hover target.")

    def _on_timer(self) -> None:
        if self._target_position is None:
            self._waiting_log_count += 1
            if self._waiting_log_count % 100 == 0:
                self.get_logger().info("Still waiting for the first odometry sample...")
            return

        command = PositionCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self._target_frame_id
        command.position.x = self._target_position[0]
        command.position.y = self._target_position[1]
        command.position.z = self._target_position[2]
        command.velocity.x = 0.0
        command.velocity.y = 0.0
        command.velocity.z = 0.0
        command.acceleration.x = 0.0
        command.acceleration.y = 0.0
        command.acceleration.z = 0.0
        command.yaw = float(self._target_yaw)
        command.yaw_dot = 0.0
        command.kx = [0.0, 0.0, 0.0]
        command.kv = [0.0, 0.0, 0.0]
        command.trajectory_id = self._trajectory_id
        command.trajectory_flag = getattr(PositionCommand, "TRAJECTORY_STATUS_READY", 1)
        self._publisher.publish(command)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the drone's current pose from odometry and keep publishing it as PositionCommand.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Reset topic used to recapture the hover target after env reset. Use empty string to disable.",
    )
    parser.add_argument("--publish-rate", type=float, default=30.0, help="Command publish rate in Hz.")
    parser.add_argument(
        "--frame-id",
        type=str,
        default="",
        help="Frame id written into PositionCommand header. Empty string reuses odometry frame_id.",
    )
    parser.add_argument(
        "--yaw-mode",
        choices=("initial", "zero"),
        default="initial",
        help="Whether to keep the captured initial yaw or force yaw=0.",
    )
    parser.add_argument("--x-offset", type=float, default=0.0, help="Optional x offset added to the captured target.")
    parser.add_argument("--y-offset", type=float, default=0.0, help="Optional y offset added to the captured target.")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Optional z offset added to the captured target.")
    return parser


def main() -> int:
    if rclpy is None or PositionCommand is None or Odometry is None:
        print("This script requires a sourced ROS 2 environment with nav_msgs, std_msgs, and quadrotor_msgs available.")
        return 1

    args = _build_argparser().parse_args()
    rclpy.init(args=None)
    node = HoverInitialPositionCommander(
        odom_topic=args.odom_topic,
        cmd_topic=args.cmd_topic,
        reset_topic=args.reset_topic,
        publish_rate_hz=args.publish_rate,
        frame_id=args.frame_id,
        yaw_mode=args.yaw_mode,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        z_offset=args.z_offset,
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
