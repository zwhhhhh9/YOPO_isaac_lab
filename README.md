# YOPO_isaac_lab

YOPO on Isaac Lab, with a project layout modeled after `Crazy_Fast`.

## What is included

- `yopo_drone/`: YOPO task package entrypoint and future task code.
- `env_tools/docker/isaaclab/`: Isaac Lab container build/run stack.
- `env_tools/autodl/`: Optional AUTODL image and launcher.
- `scripts/`: Local bootstrap and container start scripts.

## Version policy

- Isaac Lab: `v2.3.2` (latest stable release at setup time)
- Isaac Sim image: `5.1.0`
- Isaac Lab dependencies: full install (`isaaclab.sh --install`)

## Quick start

```bash
cd YOPO_isaac_lab
./scripts/init.sh
./scripts/start.sh
```

Run a python file inside container (through Isaac Sim python):

```bash
./scripts/start.sh yopo_drone/tasks/yopo/train.py --help
```

## 最简命令（初始化 YOPO Docker image + 启动 Isaac Lab GUI）

```bash
cd YOPO_isaac_lab
./scripts/init.sh
```

初始化/构建 YOPO Docker image（首次执行，后续按需重建）：

```bash
BASE_DIR=$PWD docker compose -f env_tools/docker/isaaclab/docker-compose.yml build yopo
```

启动 Isaac Lab GUI（以 quadcopter demo 为例）：

```bash
./scripts/start.sh env_tools/docker/isaaclab/IsaacLab/scripts/demos/quadcopter.py
```
