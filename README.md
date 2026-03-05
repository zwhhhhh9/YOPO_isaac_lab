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


## 最简命令（初始化 + 启动 Isaac Lab GUI）

首次初始化（只需一次）：

```bash
cd YOPO_isaac_lab && ./scripts/init.sh
```

一行启动容器并打开 Isaac Lab GUI（首次会自动构建镜像）：

```bash
cd YOPO_isaac_lab && ./scripts/start.sh --gui
```
