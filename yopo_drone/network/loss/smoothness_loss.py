from __future__ import annotations

import torch as th
import torch.nn as nn


class SmoothnessLoss(nn.Module):
    def __init__(self, rj: th.Tensor, ra: th.Tensor):
        super().__init__()
        self._rj = rj
        self._ra = ra

    def forward(self, df: th.Tensor, dp: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Args:
            dp: (batch, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
            df: (batch, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
        Returns:
            jerk_cost: (batch,)
            accel_cost: (batch,)
        """
        rj = self._rj.unsqueeze(0).expand(dp.shape[0], -1, -1)
        ra = self._ra.unsqueeze(0).expand(dp.shape[0], -1, -1)
        d_all = th.cat([df, dp], dim=2)

        dx = d_all[:, 0].unsqueeze(2)
        dy = d_all[:, 1].unsqueeze(2)
        dz = d_all[:, 2].unsqueeze(2)

        jerk_cost = dx.transpose(1, 2) @ rj @ dx + dy.transpose(1, 2) @ rj @ dy + dz.transpose(1, 2) @ rj @ dz
        accel_cost = dx.transpose(1, 2) @ ra @ dx + dy.transpose(1, 2) @ ra @ dy + dz.transpose(1, 2) @ ra @ dz
        return jerk_cost.squeeze(-1).squeeze(-1), accel_cost.squeeze(-1).squeeze(-1)
