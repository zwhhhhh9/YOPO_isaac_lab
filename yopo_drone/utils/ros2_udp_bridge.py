#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import socket
import signal
import struct
from multiprocessing import shared_memory
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class DepthSharedMemoryReader:
    _HEADER = struct.Struct("<IId")

    def __init__(self, name: str) -> None:
        self._name = name
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._last_timestamp = -1.0

    def read(self) -> Optional[tuple[np.ndarray, float]]:
        if self._shm is None:
            try:
                self._shm = shared_memory.SharedMemory(name=self._name, create=False)
            except FileNotFoundError:
                return None
        width, height, timestamp = self._HEADER.unpack_from(self._shm.buf, 0)
        if width <= 0 or height <= 0 or timestamp <= self._last_timestamp:
            return None
        payload_size = width * height * 2
        start = self._HEADER.size
        payload = self._shm.buf[start : start + payload_size]
        depth_mm = np.frombuffer(payload, dtype=np.uint16, count=width * height).copy().reshape((height, width))
        self._last_timestamp = timestamp
        return depth_mm, timestamp

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm = None


class Ros2UdpBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("yopo_ros2_udp_bridge")
        self._frame_id = args.frame_id
        self._cmd_target = (args.cmd_host, args.cmd_port)
        self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._state_socket.bind((args.state_host, args.state_port))
        self._state_socket.setblocking(False)
        self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._depth_reader = DepthSharedMemoryReader(args.depth_shm_name) if args.depth_shm_name else None

        self._odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)
        self._reset_pub = self.create_publisher(Bool, args.reset_topic, 10)
        self._goal_pub = self.create_publisher(PoseStamped, args.goal_topic, 10)
        self._depth_pub = self.create_publisher(Image, args.depth_topic, 10) if args.depth_topic else None
        self._cmd_sub = self.create_subscription(PositionCommand, args.cmd_topic, self._on_position_command, 10)

        self._poll_timer = self.create_timer(0.01, self._poll_state_socket)
        self._depth_timer = self.create_timer(1.0 / max(args.depth_rate_hz, 1e-3), self._publish_depth_from_shm)
        self.get_logger().info(
            f"ROS2 UDP bridge ready: state={args.state_host}:{args.state_port}, cmd={args.cmd_host}:{args.cmd_port}."
        )

    def _fill_stamp(self, header, timestamp: float, frame_id: str) -> None:
        sec = int(timestamp)
        nanosec = int((timestamp - sec) * 1e9)
        header.stamp.sec = sec
        header.stamp.nanosec = nanosec
        header.frame_id = frame_id or self._frame_id

    def _on_position_command(self, msg: PositionCommand) -> None:
        payload = {
            "type": "position_command",
            "stamp": float(self.get_clock().now().nanoseconds) / 1e9,
            "position": [float(msg.position.x), float(msg.position.y), float(msg.position.z)],
            "velocity": [float(msg.velocity.x), float(msg.velocity.y), float(msg.velocity.z)],
            "acceleration": [float(msg.acceleration.x), float(msg.acceleration.y), float(msg.acceleration.z)],
            "yaw": float(msg.yaw),
            "yaw_dot": float(msg.yaw_dot),
        }
        self._cmd_socket.sendto(json.dumps(payload).encode("utf-8"), self._cmd_target)

    def _poll_state_socket(self) -> None:
        while True:
            try:
                packet, _ = self._state_socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                payload = json.loads(packet.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            msg_type = payload.get("type")
            if msg_type == "odom":
                self._publish_odometry(payload)
            elif msg_type == "reset":
                msg = Bool()
                msg.data = bool(payload.get("data", False))
                self._reset_pub.publish(msg)
            elif msg_type == "goal":
                self._publish_goal(payload)

    def _publish_odometry(self, payload: dict) -> None:
        msg = Odometry()
        self._fill_stamp(msg.header, float(payload.get("stamp", 0.0)), payload.get("frame_id", self._frame_id))
        msg.child_frame_id = payload.get("child_frame_id", "drone_base")
        position = payload.get("position", [0.0, 0.0, 0.0])
        orientation = payload.get("orientation", [1.0, 0.0, 0.0, 0.0])
        linear = payload.get("linear_velocity", [0.0, 0.0, 0.0])
        angular = payload.get("angular_velocity", [0.0, 0.0, 0.0])
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, position)
        msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z = map(float, orientation)
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = map(float, linear)
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z = map(float, angular)
        msg.pose.covariance = [0.0] * 36
        msg.twist.covariance = [0.0] * 36
        self._odom_pub.publish(msg)

    def _publish_goal(self, payload: dict) -> None:
        msg = PoseStamped()
        self._fill_stamp(msg.header, float(payload.get("stamp", 0.0)), payload.get("frame_id", self._frame_id))
        position = payload.get("position", [0.0, 0.0, 0.0])
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)

    def _publish_depth_from_shm(self) -> None:
        if self._depth_pub is None or self._depth_reader is None:
            return
        data = self._depth_reader.read()
        if data is None:
            return
        depth_mm, timestamp = data
        msg = Image()
        self._fill_stamp(msg.header, timestamp, self._frame_id)
        msg.height = int(depth_mm.shape[0])
        msg.width = int(depth_mm.shape[1])
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = int(depth_mm.shape[1]) * 2
        msg.data = depth_mm.tobytes()
        self._depth_pub.publish(msg)

    def close(self) -> None:
        if self._depth_reader is not None:
            self._depth_reader.close()
        self._state_socket.close()
        self._cmd_socket.close()
        self.destroy_node()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS2 sidecar bridge for eval_ego UDP transport.")
    parser.add_argument("--frame-id", type=str, default="world")
    parser.add_argument("--depth-topic", type=str, default="/drone_0_depth")
    parser.add_argument("--depth-rate-hz", type=float, default=15.0)
    parser.add_argument("--depth-shm-name", type=str, default="depth_image_shm")
    parser.add_argument("--odom-topic", type=str, default="/drone_0_odometry")
    parser.add_argument("--cmd-topic", type=str, default="/drone_0_planning/pos_cmd")
    parser.add_argument("--reset-topic", type=str, default="/drone_0_reset")
    parser.add_argument("--goal-topic", type=str, default="/move_base_simple/goal")
    parser.add_argument("--cmd-host", type=str, default="127.0.0.1")
    parser.add_argument("--cmd-port", type=int, default=15000)
    parser.add_argument("--state-host", type=str, default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=15001)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    rclpy.init(args=None)
    node = Ros2UdpBridge(args)

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
        if "context is not valid" not in str(exc):
            raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        with contextlib.suppress(Exception):
            node.close()
        if rclpy.ok():
            with contextlib.suppress(Exception):
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
