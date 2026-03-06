# YOPO_isaac_lab 使用命令

## 1. 初始化并配置 Docker 环境

```bash
cd YOPO_isaac_lab
./scripts/init.sh
```

说明：首次执行会准备 IsaacLab 源码；后续可重复执行用于校验或更新初始化状态。

## 2. 测试 Isaac Lab GUI

```bash
cd YOPO_isaac_lab
./scripts/start.sh --gui
```

说明：该命令会拉起容器并进入 Isaac Lab GUI。若报 `DISPLAY is not set`，请先配置本机图形显示环境再执行。

## 3. 测试环境配置 Python 程序（drone_env_editor.py）

1) 查看程序参数（快速验证 Python 运行环境可用）：

```bash
cd YOPO_isaac_lab
./scripts/start.sh --editor --help
```

2) GUI 模式下运行一次环境编辑测试：

```bash
cd YOPO_isaac_lab
./scripts/start.sh --editor --random-obstacles 2
```

3) 无界面执行并导出 USD（用于配置测试留档）：

```bash
cd YOPO_isaac_lab
./scripts/start.sh --editor \
  --headless \
  --close-after-build \
  --output-usd /workspace/isaaclab/logs/drone_env_test.usd \
  --random-obstacles 2
```

4) 在宿主机检查导出文件：

```bash
cd YOPO_isaac_lab
ls -lh logs/drone_env_test.usd
file logs/drone_env_test.usd
```
