# YOPO Isaac Lab

Docker-based workspace for running Isaac Lab, the YOPO environment editor, and ROS 2 connected drone evaluation scripts.

## Repository Layout

```text
scripts/                     # Entry scripts (init, start)
env_tools/docker/isaaclab/   # Dockerfile, compose, Isaac Lab source checkout
yopo_drone/                  # YOPO runtime, env editor, eval controller, ROS2 bridge helpers
ros2_ws/                     # ROS 2 workspace for custom messages (quadrotor_msgs)
assets/                      # Robot and scene assets
logs/                        # Runtime logs mounted from container
```

## Prerequisites

- Linux with NVIDIA GPU and working NVIDIA drivers
- Docker + Docker Compose plugin
- X11 desktop session for GUI runs (`DISPLAY` must be available)

## Quick Start

### 1) Initialize the workspace

This prepares the local Isaac Lab source tree under `env_tools/docker/isaaclab/IsaacLab`.

```bash
cd YOPO_isaac_lab
./scripts/init.sh
```

### 2) Rebuild the image after ROS 2 changes

The container now installs ROS 2 Jazzy and builds `ros2_ws` on demand when you launch ROS nodes.

```bash
docker compose -f env_tools/docker/isaaclab/docker-compose.yml build yopo
```

### 3) Test Isaac Lab GUI startup

This launches Isaac Lab GUI directly in the container.

```bash
./scripts/start.sh --gui
```

### 4) Test the env editor Python program

The env editor entry script is `yopo_drone/env/drone_env_editor.py`.

Run in GUI mode:

```bash
./scripts/start.sh --env_editor
```

Run headless smoke test (build scene and exit):

```bash
./scripts/start.sh --env_editor --headless --close-after-build
```

Show env editor arguments/help:

```bash
./scripts/start.sh --env_editor --help
```

## ROS 2 Hover Workflow

### 1) Start `eval_ego.py`

`eval_ego.py` now does two things:
- Runs the PX4 controller inside Isaac Lab
- When direct `rclpy` is unavailable in Isaac Python, automatically starts a ROS 2 sidecar bridge with system Python

Run it headless:

```bash
./scripts/start.sh yopo_drone/utils/eval_ego.py --headless --num_envs 1
```

If you want the planner/PX4 stack to first initialize the stage with the same world primitives as `drone_env_editor.py` (for example, to keep the ground plane visible), use the task wrapper:

```bash
./scripts/start.sh yopo_drone/tasks/editor_scene_eval_ego.py --headless --num_envs 1 --reset_log_count 0
```

### 2) Start the hover command node

In another terminal, launch the ROS 2 node with system Python inside the same image:

```bash
./scripts/start.sh --ros2-node yopo_drone/tasks/hover_initial_position.py
```

This node listens to `/drone_0_odometry`, captures the current initial pose, and keeps publishing it to `/drone_0_planning/pos_cmd` so the PX4 controller can hold position.

### 3) ROS 2 transport summary

- `eval_ego.py` publishes depth through shared memory and odometry/reset/goal through a sidecar ROS 2 bridge
- `hover_initial_position.py` is a normal ROS 2 node using `quadrotor_msgs/PositionCommand`
- `scripts/start.sh --ros2-node ...` uses system `/usr/bin/python3` with sourced ROS 2 environment

## Useful Commands

Stop all running YOPO Isaac Lab containers:

```bash
./scripts/start.sh --stop-all
```

Show launcher help:

```bash
./scripts/start.sh --help
```

## Notes

- `scripts/start.sh` auto-creates required Docker volumes and normalizes `ros2_ws/{build,install,log}` ownership back to the host user.
- On exit, the launched compose project is automatically cleaned up.
- If GUI does not appear, verify X11 access and `DISPLAY` on host.
- `ros2_ws/src/quadrotor_msgs` is the single source of truth for the custom ROS 2 message package.
