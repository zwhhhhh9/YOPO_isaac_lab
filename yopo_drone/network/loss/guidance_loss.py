from __future__ import annotations

import torch as th
import torch.nn as nn
import torch.nn.functional as f

from .config import ensure_config


class GuidanceLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.cfg = ensure_config(config)
        self.goal_length = float(self.cfg["goal_length"])
        self.vel_dir_weight = float(self.cfg.get("wgv", 0.0))

    def forward(self, df: th.Tensor, dp: th.Tensor, goal: th.Tensor) -> th.Tensor:
        """
        Args:
            dp: (batch, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
            df: (batch, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
            goal: (batch, 3)
        Returns:
            guidance_loss: (batch,)
        """
        cur_pos = df[:, :, 0]
        end_pos = dp[:, :, 0]
        end_vel = dp[:, :, 1]

        traj_dir = end_pos - cur_pos
        goal_dir = goal - cur_pos

        guidance_loss = self.similarity_loss(traj_dir, goal_dir)
        if self.vel_dir_weight > 0.0:
            guidance_loss = guidance_loss + self.vel_dir_weight * self.derivative_similarity_loss(end_vel, goal_dir)
        return guidance_loss

    def distance_loss(self, traj_dir: th.Tensor, goal_dir: th.Tensor) -> th.Tensor:
        l1_distance = f.smooth_l1_loss(traj_dir, goal_dir, reduction="none")
        return l1_distance.sum(dim=1)

    def similarity_loss(self, traj_dir: th.Tensor, goal_dir: th.Tensor) -> th.Tensor:
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)

        traj_along = (traj_dir * goal_dir_norm).sum(dim=1)
        goal_length = goal_dir.norm(dim=1)

        parallel_diff = f.smooth_l1_loss(goal_length, traj_along, reduction="none")

        traj_perp = traj_dir - traj_along.unsqueeze(1) * goal_dir_norm
        perp_diff = traj_perp.norm(dim=1)

        perp_weight = 0.5
        return parallel_diff + perp_weight * perp_diff

    def derivative_similarity_loss(self, derivative: th.Tensor, goal_dir: th.Tensor) -> th.Tensor:
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)
        derivative_norm = derivative / (derivative.norm(dim=1, keepdim=True) + 1e-8)
        similarity = (derivative_norm * goal_dir_norm).sum(dim=1)
        return 1.0 - similarity
