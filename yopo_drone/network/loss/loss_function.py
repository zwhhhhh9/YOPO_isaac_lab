from __future__ import annotations

import math

import torch as th
import torch.nn as nn

from .config import ensure_config
from .guidance_loss import GuidanceLoss
from .safety_loss import SafetyLoss
from .smoothness_loss import SmoothnessLoss


class YOPOLoss(nn.Module):
    def __init__(self, config=None, *, dataset_root=None):
        """
        Compute YOPO trajectory costs:
        smoothness, safety, goal guidance, and acceleration regularization.
        """
        super().__init__()
        self.cfg = ensure_config(config)
        self.sgm_time = float(self.cfg["sgm_time"])
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        _, _, l_matrix, r_jerk, r_acc = self.qp_generation()
        self._l = l_matrix.to(self.device)
        self._rj = r_jerk.to(self.device)
        self._ra = r_acc.to(self.device)

        self.denormalize_weight()
        self.smoothness_loss = SmoothnessLoss(self._rj, self._ra)
        self.safety_loss = SafetyLoss(self._l, config=self.cfg, dataset_root=dataset_root)
        self.goal_loss = GuidanceLoss(self.cfg)

        print("------ Actual Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print("-------------------------")

    def qp_generation(self) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        a_matrix = th.zeros((6, 6))
        for derivative_order in range(3):
            a_matrix[2 * derivative_order, derivative_order] = math.factorial(derivative_order)
            for coefficient_order in range(derivative_order, 6):
                a_matrix[2 * derivative_order + 1, coefficient_order] = (
                    math.factorial(coefficient_order)
                    / math.factorial(coefficient_order - derivative_order)
                    * (self.sgm_time ** (coefficient_order - derivative_order))
                )

        h_matrix = th.zeros((6, 6))
        for row in range(3, 6):
            for col in range(3, 6):
                h_matrix[row, col] = (
                    row
                    * (row - 1)
                    * (row - 2)
                    * col
                    * (col - 1)
                    * (col - 2)
                    / (row + col - 5)
                    * (self.sgm_time ** (row + col - 5))
                )

        q_matrix = th.zeros((6, 6))
        for row in range(2, 6):
            for col in range(2, 6):
                q_matrix[row, col] = (
                    (row * (row - 1))
                    * (col * (col - 1))
                    / (row + col - 3)
                    * (self.sgm_time ** (row + col - 3))
                )

        return self.stack_opt_dep(a_matrix, h_matrix, q_matrix)

    def stack_opt_dep(
        self,
        a_matrix: th.Tensor,
        h_matrix: th.Tensor,
        q_matrix: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        ct = th.zeros((6, 6))
        ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1

        c_matrix = ct.transpose(0, 1)
        b_matrix = th.inverse(a_matrix)
        b_transpose = b_matrix.transpose(0, 1)

        l_matrix = b_matrix @ ct
        r_jerk = c_matrix @ b_transpose @ h_matrix @ b_matrix @ ct
        r_acc = c_matrix @ b_transpose @ q_matrix @ b_matrix @ ct
        return c_matrix, b_matrix, l_matrix, r_jerk, r_acc

    def denormalize_weight(self) -> None:
        vel_scale = float(self.cfg["vel_max_train"]) / 1.0
        self.smoothness_weight = float(self.cfg["ws"]) / vel_scale**5
        self.accele_weight = float(self.cfg["wa"]) / vel_scale**3
        self.safety_weight = float(self.cfg["wc"])
        self.goal_weight = float(self.cfg["wg"])

    def forward(
        self,
        state: th.Tensor,
        prediction: th.Tensor,
        goal: th.Tensor,
        map_id: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Args:
            prediction: (batch, 3, 3) -> [px, py, pz; vx, vy, vz; ax, ay, az]
            state: (batch, 3, 3) -> [px, py, pz; vx, vy, vz; ax, ay, az]
            goal: (batch, 3)
            map_id: (batch,) or broadcastable to batch
        Returns:
            smoothness_cost, safety_cost, goal_cost, acceleration_cost
        """
        df = state.permute(0, 2, 1)
        dp = prediction.permute(0, 2, 1)

        smoothness_cost, acceleration_cost = self.smoothness_loss(df, dp)
        safety_cost = self.safety_loss(df, dp, map_id)
        goal_cost = self.goal_loss(df, dp, goal)

        return (
            self.smoothness_weight * smoothness_cost,
            self.safety_weight * safety_cost,
            self.goal_weight * goal_cost,
            self.accele_weight * acceleration_cost,
        )
