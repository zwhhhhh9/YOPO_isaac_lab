"""Training loop for the migrated YOPO model."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader

from yopo_drone.network.loss import YOPOLoss
from yopo_drone.network.models.common import make_config, state_body2world
from yopo_drone.network.models.train.yopo_dataset import YopoDataset
from yopo_drone.network.models.train.yopo_train_model import YopoTrainModel


def configure_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _create_checkpoint_run_dir(checkpoint_root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    candidate = checkpoint_root / f"checkpoint_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = checkpoint_root / f"checkpoint_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _normalize_checkpoint_name(checkpoint_name: str | Path) -> Path:
    normalized = Path(str(checkpoint_name).strip()).expanduser()
    if normalized.name in {"", "."}:
        raise ValueError("checkpoint_name must include a filename.")
    if normalized.suffix == "":
        normalized = normalized.with_suffix(".pth")
    return normalized


def _best_checkpoint_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")


def _history_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_history.json")


def _summary_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_summary.json")


@dataclass
class TrainingSummary:
    dataset_dir: str
    checkpoint_path: str
    best_checkpoint_path: str
    history_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    device: str
    train_samples: int
    valid_samples: int
    duration_sec: float
    best_val_total_loss: float
    final_val_total_loss: float
    config: dict[str, object]


class YopoTrainer:
    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        checkpoint_dir: str | Path,
        checkpoint_name: str,
        epochs: int = 50,
        learning_rate: float = 1.5e-4,
        batch_size: int = 16,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        loss_weight: tuple[float, float] = (1.0, 1.0),
        compact_backbone: bool = False,
        seed: int = 0,
        device: str | None = None,
        goal_sampling_mode: str = "gaussian_forward",
        goal_yaw_uniform_min_deg: float = -180.0,
        goal_yaw_uniform_max_deg: float = 180.0,
        goal_pitch_uniform_min_deg: float = -30.0,
        goal_pitch_uniform_max_deg: float = 30.0,
        safety_weight: float = 1.0,
        safety_distance_margin: float = 1.2,
        safety_decay_radius: float = 0.6,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = _normalize_checkpoint_name(checkpoint_name)
        if self.checkpoint_name.is_absolute():
            self.final_checkpoint_path = self.checkpoint_name.resolve()
        else:
            self.checkpoint_run_dir = _create_checkpoint_run_dir(self.checkpoint_dir)
            self.final_checkpoint_path = self.checkpoint_run_dir / self.checkpoint_name
        self.checkpoint_run_dir = self.final_checkpoint_path.parent
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_ratio = float(val_ratio)
        self.loss_weight = (float(loss_weight[0]), float(loss_weight[1]))
        self.seed = int(seed)

        configure_random_seed(self.seed)

        self.device = self._resolve_device(device)
        self.config = make_config(
            {
                "dataset_path": str(self.dataset_dir),
                "goal_sampling_mode": str(goal_sampling_mode),
                "goal_yaw_uniform_min_deg": float(goal_yaw_uniform_min_deg),
                "goal_yaw_uniform_max_deg": float(goal_yaw_uniform_max_deg),
                "goal_pitch_uniform_min_deg": float(goal_pitch_uniform_min_deg),
                "goal_pitch_uniform_max_deg": float(goal_pitch_uniform_max_deg),
                "wc": float(safety_weight),
                "d0": float(safety_distance_margin),
                "r": float(safety_decay_radius),
            },
            train=True,
        )
        self.traj_num = int(self.config["traj_num"])
        self.max_grad_norm = 0.1

        self.model = YopoTrainModel(
            compact_backbone=compact_backbone,
            config=self.config,
            device=self.device,
        )
        self.yopo_loss = YOPOLoss(config=self.config.to_dict(), dataset_root=self.dataset_dir)

        optimizer_kwargs = {"lr": self.learning_rate}
        if self.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        train_dataset = YopoDataset(self.dataset_dir, mode="train", val_ratio=self.val_ratio, split_seed=self.seed, config=self.config)
        valid_dataset = YopoDataset(self.dataset_dir, mode="valid", val_ratio=self.val_ratio, split_seed=self.seed, config=self.config)
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=_seed_worker,
            generator=generator,
            persistent_workers=self.num_workers > 0,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=_seed_worker,
            persistent_workers=self.num_workers > 0,
        )

        self.train_samples = len(train_dataset)
        self.valid_samples = len(valid_dataset)
        self.history: list[dict[str, float | int]] = []
        self.best_val_total_loss = math.inf
        self.final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = _best_checkpoint_path(self.final_checkpoint_path)
        self.history_path = _history_path(self.final_checkpoint_path)
        self.summary_path = _summary_path(self.final_checkpoint_path)

    def _resolve_device(self, requested: str | None) -> torch.device:
        if requested and requested != "auto":
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_and_compute_loss(
        self,
        depth: torch.Tensor,
        pos: torch.Tensor,
        rot: torch.Tensor,
        obs_b: torch.Tensor,
        map_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = depth.to(self.device, non_blocking=True)
        pos = pos.to(self.device, non_blocking=True)
        rot = rot.to(self.device, non_blocking=True)
        obs_b = obs_b.to(self.device, non_blocking=True)
        map_id = map_id.to(self.device, non_blocking=True)

        goal_w, start_vel_w, start_acc_w = state_body2world(
            pos,
            rot,
            obs_b[:, 6:9],
            obs_b[:, 0:3],
            obs_b[:, 3:6],
        )
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

        endstate, score = self.model.inference(depth, obs_b)

        batch_size = depth.shape[0]
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(batch_size * self.traj_num, 9)
        score_flat = score.reshape(batch_size * self.traj_num)

        pos_expanded = pos.repeat_interleave(self.traj_num, dim=0)
        rot_expanded = rot.repeat_interleave(self.traj_num, dim=0)
        start_state_w = start_state_w.repeat_interleave(self.traj_num, dim=0)
        goal_w = goal_w.repeat_interleave(self.traj_num, dim=0)

        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded,
            rot_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9],
        )
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

        smooth_cost, safety_cost, goal_cost, acc_cost = self.yopo_loss(start_state_w, end_state_w, goal_w, map_id)
        total_cost = smooth_cost + safety_cost + goal_cost + acc_cost
        trajectory_loss = total_cost.mean()
        score_label = total_cost.detach()
        score_loss = f.smooth_l1_loss(score_flat, score_label)
        return (
            trajectory_loss,
            score_loss,
            smooth_cost.mean(),
            safety_cost.mean(),
            goal_cost.mean(),
            acc_cost.mean(),
        )

    def _run_one_epoch(self, epoch: int, *, training: bool) -> dict[str, float]:
        loader = self.train_loader if training else self.valid_loader
        self.model.train(training)

        totals = {
            "traj_loss": 0.0,
            "score_loss": 0.0,
            "smooth_loss": 0.0,
            "safety_loss": 0.0,
            "goal_loss": 0.0,
            "accel_loss": 0.0,
            "total_loss": 0.0,
        }
        steps = 0

        for depth, pos, rot, obs_b, map_id in loader:
            steps += 1
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            autocast_enabled = self.device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                trajectory_loss, score_loss, smooth_cost, safety_cost, goal_cost, acc_cost = self.forward_and_compute_loss(
                    depth, pos, rot, obs_b, map_id
                )
                total_loss = self.loss_weight[0] * trajectory_loss + self.loss_weight[1] * score_loss

            if training:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            totals["traj_loss"] += float(trajectory_loss.detach().cpu())
            totals["score_loss"] += float(score_loss.detach().cpu())
            totals["smooth_loss"] += float(smooth_cost.detach().cpu())
            totals["safety_loss"] += float(safety_cost.detach().cpu())
            totals["goal_loss"] += float(goal_cost.detach().cpu())
            totals["accel_loss"] += float(acc_cost.detach().cpu())
            totals["total_loss"] += float(total_loss.detach().cpu())

        if steps == 0:
            raise RuntimeError(f"No batches were produced for {'train' if training else 'valid'} loader.")

        for key in totals:
            totals[key] /= steps
        prefix = "train" if training else "valid"
        return {f"{prefix}_{key}": value for key, value in totals.items()}

    def train(self) -> TrainingSummary:
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            train_metrics = self._run_one_epoch(epoch, training=True)
            with torch.inference_mode():
                valid_metrics = self._run_one_epoch(epoch, training=False)

            epoch_metrics: dict[str, float | int] = {"epoch": epoch, **train_metrics, **valid_metrics}
            self.history.append(epoch_metrics)

            val_total_loss = float(valid_metrics["valid_total_loss"])
            print(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"train_total={train_metrics['train_total_loss']:.6f} "
                f"val_total={val_total_loss:.6f} "
                f"traj={valid_metrics['valid_traj_loss']:.6f} "
                f"score={valid_metrics['valid_score_loss']:.6f}",
                flush=True,
            )

            if val_total_loss < self.best_val_total_loss:
                self.best_val_total_loss = val_total_loss
                torch.save(self.model.state_dict(), self.best_checkpoint_path)

        torch.save(self.model.state_dict(), self.final_checkpoint_path)
        self._write_history()

        duration = time.time() - start_time
        final_val_total = float(self.history[-1]["valid_total_loss"])
        summary = TrainingSummary(
            dataset_dir=str(self.dataset_dir),
            checkpoint_path=str(self.final_checkpoint_path),
            best_checkpoint_path=str(self.best_checkpoint_path),
            history_path=str(self.history_path),
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=str(self.device),
            train_samples=self.train_samples,
            valid_samples=self.valid_samples,
            duration_sec=duration,
            best_val_total_loss=float(self.best_val_total_loss),
            final_val_total_loss=final_val_total,
            config=self.config.to_dict(),
        )
        self.summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return summary

    def _write_history(self) -> None:
        payload = {
            "dataset_dir": str(self.dataset_dir),
            "checkpoint_path": str(self.final_checkpoint_path),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "epochs": self.epochs,
            "config": self.config.to_dict(),
            "history": self.history,
        }
        self.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
