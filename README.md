# YOPO Isaac Lab

This repository is mainly used for 4 tasks:

1. Initialize the Isaac Lab environment
2. Collect YOPO training data
3. Train a YOPO model
4. Run closed-loop inference and inspect the flight behavior

## Requirements

- Linux
- NVIDIA GPU with working drivers
- Docker with the Docker Compose plugin
- Network access for the first initialization
- A valid X11 `DISPLAY` on the host if you want to use the GUI

## 1. Initialization

Initialize the Isaac Lab source tree first, then build the runtime image.

```bash
cd YOPO_isaac_lab
./scripts/init.sh
docker compose -f env_tools/docker/isaaclab/docker-compose.yml build yopo
```

## 2. Data Collection

Dataset collection entrypoint:

```bash
./scripts/start.sh yopo_drone/network/collect_dataset/collect_yopo_dataset.py --headless
```

A more typical collection example:

```bash
./scripts/start.sh yopo_drone/network/collect_dataset/collect_yopo_dataset.py \
  --headless \
  --env_num 10 \
  --image_num 10000 \
  --dataset_timestamp 20260324_my_dataset
```

Collected data will be written to:

```text
yopo_drone/network/data_train/<dataset_timestamp>/
```

The directory mainly contains:

- `img/`: depth images
- `pose.csv`: poses
- `pointcloud.ply`: point cloud
- `metadata.json`: collection parameters for this run

Common arguments:

- `--env_num`: number of random maps to generate
- `--image_num`: number of images to collect per map
- `--dataset_timestamp`: output directory name
- `--safe_dist`: minimum clearance between the camera pose and obstacles

## 3. Training

Training entrypoint:

```bash
./scripts/start.sh yopo_drone/network/models/train/train_yopo.py \
  --dataset-dir yopo_drone/network/data_train/<dataset_timestamp>
```

A more typical training example:

```bash
./scripts/start.sh yopo_drone/network/models/train/train_yopo.py \
  --dataset-dir yopo_drone/network/data_train/20260324_my_dataset \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1.5e-4
```

Training outputs will be written to:

```text
yopo_drone/network/checkpoint/checkpoint_<timestamp>/
```

The checkpoint directory typically contains:

- `*.pth`: final checkpoint
- `*_best.pth`: best checkpoint on validation loss
- `*_history.json`: training history
- `*_summary.json`: training summary and inference config recovery data

## 4. Inference

### 4.1 View Flight Behavior in the GUI

If you want to quickly inspect the flight behavior with the default configured model:

```bash
./scripts/start.sh yopo_drone/network/models/test/yopo_policy_gui.py --num_envs 1
```

### 4.2 Use Your Own Checkpoint and Dataset

```bash
./scripts/start.sh yopo_drone/network/models/test/yopo_policy_gui.py \
  --checkpoint yopo_drone/network/checkpoint/checkpoint_xxx/epoch50_best.pth \
  --dataset_dir yopo_drone/network/data_train/<dataset_timestamp> \
  --num_envs 1
```

### 4.3 Headless Inference

```bash
./scripts/start.sh yopo_drone/tasks/editor_scene_eval_ego.py \
  --yopo_policy \
  --headless \
  --num_envs 1 \
  --yopo_policy_checkpoint yopo_drone/network/checkpoint/checkpoint_xxx/epoch50_best.pth \
  --yopo_policy_dataset_dir yopo_drone/network/data_train/<dataset_timestamp>
```

Inference telemetry CSV files will be written to:

```text
yopo_drone/logs/
```

## Recommended Workflow

```bash
./scripts/init.sh
docker compose -f env_tools/docker/isaaclab/docker-compose.yml build yopo

./scripts/start.sh yopo_drone/network/collect_dataset/collect_yopo_dataset.py \
  --headless \
  --env_num 10 \
  --image_num 10000 \
  --dataset_timestamp 20260324_my_dataset

./scripts/start.sh yopo_drone/network/models/train/train_yopo.py \
  --dataset-dir yopo_drone/network/data_train/20260324_my_dataset \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1.5e-4

./scripts/start.sh yopo_drone/network/models/test/yopo_policy_gui.py \
  --checkpoint yopo_drone/network/checkpoint/checkpoint_xxx/epoch50_best.pth \
  --dataset_dir yopo_drone/network/data_train/20260324_my_dataset \
  --num_envs 1
```
