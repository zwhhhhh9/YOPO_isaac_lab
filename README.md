# YOPO Isaac Lab

Docker-based workspace for running Isaac Lab and the YOPO environment editor.

## Repository Layout

```text
scripts/                 # Entry scripts (init, start)
env_tools/docker/isaaclab/  # Dockerfile, compose, Isaac Lab source checkout
yopo_drone/              # YOPO runtime and env editor program
assets/                  # Robot and scene assets
logs/                    # Runtime logs mounted from container
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

### 2) Test Isaac Lab GUI startup

This launches Isaac Lab GUI directly in the container.

```bash
./scripts/start.sh --gui
```

### 3) Test the env_editor Python program

The env editor entry script is:
`yopo_drone/env/drone_env_editor.py`

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

- `scripts/start.sh` auto-creates required Docker volumes.
- On exit, the launched compose project is automatically cleaned up.
- If GUI does not appear, verify X11 access and `DISPLAY` on host.
