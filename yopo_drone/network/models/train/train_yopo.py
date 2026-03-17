#!/usr/bin/env python3
"""Train the migrated YOPO model with the current repository layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yopo_drone.network.models.train.yopo_trainer import YopoTrainer


def _default_checkpoint_name(*, epochs: int, goal_sampling_mode: str) -> str:
    stem = f"epoch{int(epochs)}"
    if goal_sampling_mode == "uniform_full_yaw":
        stem += "_full_yaw"
    elif goal_sampling_mode == "uniform_box":
        stem += "_uniform_box"
    else:
        stem += "_gaussian_forward"
    return f"{stem}.pth"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOPO on a collected Isaac Lab dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="yopo_drone/network/data_train/20260317_022500_recollect_goalfix_1map",
        help="Dataset directory containing pose.csv/img/pointcloud.ply.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="yopo_drone/network/checkpoint",
        help="Directory for output checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Final checkpoint filename. Defaults to a mode-aware epoch filename inside a timestamped checkpoint folder.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="Device string, e.g. auto/cpu/cuda:0.")
    parser.add_argument("--compact-backbone", action="store_true", help="Use the smaller ResNet14 backbone.")
    parser.add_argument(
        "--goal-sampling-mode",
        type=str,
        default="uniform_full_yaw",
        choices=("gaussian_forward", "uniform_full_yaw", "uniform_box"),
        help="Training goal sampling mode. Default uses full-yaw sampling to cover arbitrary mission-goal directions.",
    )
    parser.add_argument(
        "--goal-yaw-uniform-range",
        type=float,
        nargs=2,
        metavar=("MIN_DEG", "MAX_DEG"),
        default=(-180.0, 180.0),
        help="Uniform yaw range in degrees used by uniform goal sampling modes.",
    )
    parser.add_argument(
        "--goal-pitch-uniform-range",
        type=float,
        nargs=2,
        metavar=("MIN_DEG", "MAX_DEG"),
        default=(-30.0, 30.0),
        help="Uniform pitch range in degrees used by the uniform_box goal sampling mode.",
    )
    parser.add_argument(
        "--safety-weight",
        type=float,
        default=1.5,
        help="Safety-loss weight wc. Larger values bias the model toward safer trajectories.",
    )
    parser.add_argument(
        "--safety-d0",
        type=float,
        default=1.6,
        help="Safety-loss clearance target d0 in exp(-(d - d0) / r). Larger values prefer more obstacle clearance.",
    )
    parser.add_argument(
        "--safety-r",
        type=float,
        default=0.8,
        help="Safety-loss decay radius r in exp(-(d - d0) / r). Larger values keep obstacle penalties active farther away.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    checkpoint_name = args.checkpoint_name or _default_checkpoint_name(
        epochs=int(args.epochs),
        goal_sampling_mode=str(args.goal_sampling_mode),
    )
    trainer = YopoTrainer(
        dataset_dir=Path(args.dataset_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        checkpoint_name=checkpoint_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        compact_backbone=bool(args.compact_backbone),
        seed=args.seed,
        device=args.device,
        goal_sampling_mode=args.goal_sampling_mode,
        goal_yaw_uniform_min_deg=float(args.goal_yaw_uniform_range[0]),
        goal_yaw_uniform_max_deg=float(args.goal_yaw_uniform_range[1]),
        goal_pitch_uniform_min_deg=float(args.goal_pitch_uniform_range[0]),
        goal_pitch_uniform_max_deg=float(args.goal_pitch_uniform_range[1]),
        safety_weight=args.safety_weight,
        safety_distance_margin=args.safety_d0,
        safety_decay_radius=args.safety_r,
    )
    summary = trainer.train()
    print(f"Final checkpoint: {summary.checkpoint_path}", flush=True)
    print(f"History: {summary.history_path}", flush=True)
    print(f"Best val total loss: {summary.best_val_total_loss:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
