# YOPO Isaac Lab

Docker-based workspace for Isaac Lab, the YOPO drone scene editor, and the PX4-style hover / evaluation pipeline.

## What This Repository Contains

- Isaac Lab source checkout and Docker environment
- YOPO drone environment editor and evaluation entry points
- A URDF-driven drone model plus matching PX4 controller parameters
- A ROS 2 workspace for custom messages such as `quadrotor_msgs/PositionCommand`

## Key Entry Points

| Path | Purpose |
| --- | --- |
| `scripts/init.sh` | Initialize or refresh the local Isaac Lab source tree |
| `scripts/start.sh` | Main launcher for GUI, env editor, eval scripts, and ROS 2 nodes |
| `yopo_drone/env/drone_env_editor.py` | Scene editor / stage construction entry |
| `yopo_drone/tasks/editor_scene_eval_ego.py` | Preferred hover-eval entry with editor-style stage initialization |
| `yopo_drone/utils/eval_ego.py` | Core PX4-style controller / telemetry / ROS bridge runner |
| `yopo_drone/tasks/hover_initial_position.py` | Optional ROS 2 node that captures odometry and republishes a hover target |
| `assets/robot./robot.urdf` | Drone URDF used by the runtime |
| `assets/robot./px4_params.json` | PX4 controller parameters matched to the URDF model |

## Repository Layout

```text
scripts/                     Launcher scripts
env_tools/docker/isaaclab/   Docker setup and Isaac Lab checkout
yopo_drone/                  Editor, eval, controller, ROS bridge helpers
yopo_drone/logs/             Drone telemetry and CSV data collected from hover/eval runs
ros2_ws/                     ROS 2 workspace for custom message packages
assets/                      Robot and scene assets
logs/                        Host-side runtime logs
```

## Requirements

- Linux with an NVIDIA GPU and working NVIDIA drivers
- Docker with the Docker Compose plugin
- X11 desktop session for GUI runs (`DISPLAY` must be available)
- Network access for the first `init.sh` or image rebuild when needed

## Setup

### 1. Initialize Isaac Lab

```bash
cd YOPO_isaac_lab
./scripts/init.sh
```

### 2. Build or rebuild the Docker image

Run this after Dockerfile changes, ROS 2 workspace changes, or whenever dependencies inside the image changed.

```bash
docker compose -f env_tools/docker/isaaclab/docker-compose.yml build yopo
```

### 3. Smoke-test the base Isaac Lab GUI

```bash
./scripts/start.sh --gui
```

## Common Workflows

### Open the YOPO env editor

GUI mode:

```bash
./scripts/start.sh --env_editor
```

Headless build-and-exit smoke test:

```bash
./scripts/start.sh --env_editor --headless --close-after-build
```

Show env editor arguments:

```bash
./scripts/start.sh --env_editor --help
```

### Run the drone hover / evaluation scene

This is the recommended entry point for drone hover checks. It first initializes the stage with the same helpers used by `drone_env_editor.py`, then launches `eval_ego.py`.

GUI mode:

```bash
./scripts/start.sh yopo_drone/tasks/editor_scene_eval_ego.py --num_envs 1
```

Headless mode:

```bash
./scripts/start.sh yopo_drone/tasks/editor_scene_eval_ego.py --headless --num_envs 1
```

Headless mode with automatic telemetry export:

```bash
./scripts/start.sh yopo_drone/tasks/editor_scene_eval_ego.py --headless --num_envs 1
```

Each hover / eval run automatically collects drone telemetry data into `yopo_drone/logs/`.
The per-run CSV filename uses the form `hvoer_YYYYMMDD_HHMMSS.csv`, for example
`hvoer_20260310_142530.csv`.

### Run `eval_ego.py` directly

Use this only when you specifically want the raw eval runner without the editor-style scene wrapper.

```bash
./scripts/start.sh yopo_drone/utils/eval_ego.py --headless --num_envs 1
```

### Start the optional ROS 2 hover command node

`eval_ego.py` can already hold position internally when no planner commands arrive. This node is only needed if you want an external ROS 2 `PositionCommand` publisher that captures the current odometry pose and keeps sending it.

```bash
./scripts/start.sh --ros2-node yopo_drone/tasks/hover_initial_position.py
```

Useful variants:

```bash
./scripts/start.sh --ros2-node yopo_drone/tasks/hover_initial_position.py --yaw-mode zero
./scripts/start.sh --ros2-node yopo_drone/tasks/hover_initial_position.py --z-offset 0.2
```

## Control Model and Asset Mapping

- `assets/robot./robot.urdf` is the drone model used by the editor and evaluation runtime.
- `assets/robot./px4_params.json` stores the PX4-style controller tuning matched to that URDF.
- `yopo_drone/utils/eval_ego.py` loads the URDF-derived rigid-body parameters, applies the PX4 tuning, and runs the controller.
- `yopo_drone/utils/px4_controller.py` contains the PX4-style attitude / rate control implementation.

If you change the drone geometry, mass properties, or rotor layout, update the URDF and then re-check the matching controller parameters in `assets/robot./px4_params.json`.

## ROS 2 Notes

- `scripts/start.sh --ros2-node ...` uses system `/usr/bin/python3` inside the container with ROS 2 sourced.
- When direct `rclpy` import is unavailable inside Isaac Python, `eval_ego.py` automatically starts a ROS 2 UDP sidecar bridge.
- The custom message package source of truth is `ros2_ws/src/quadrotor_msgs`.

## Useful Commands

Show launcher help:

```bash
./scripts/start.sh --help
```

Stop all running YOPO Isaac Lab containers:

```bash
./scripts/start.sh --stop-all
```

## Runtime Notes

- `scripts/start.sh` creates a fresh disposable Docker Compose project for each run.
- On exit, that Compose project is cleaned up automatically.
- The launcher auto-creates shared Docker volumes used by Isaac Sim caches and logs.
- `ros2_ws/build`, `ros2_ws/install`, and `ros2_ws/log` are built on demand and ownership is normalized back to the host user.
- If a GUI window does not appear, first check `DISPLAY`, X11 permissions, and host-side `xhost` access.
