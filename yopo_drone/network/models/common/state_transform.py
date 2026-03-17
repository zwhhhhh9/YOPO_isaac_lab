"""State transforms between YOPO frames."""

from __future__ import annotations

import numpy as np
import torch

from .config import Config, make_config
from .primitive import LatticePrimitive


class StateTransform:
    def __init__(self, config: Config | None = None):
        self.cfg = make_config(config)
        self.lattice_primitive = LatticePrimitive(self.cfg)
        self.goal_length = float(self.cfg["goal_length"])

    def pred_to_endstate(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
        Transform model predictions to body-frame end states.

        Args:
            endstate_pred: [B, 9, V, H]
        Returns:
            [B, 9, V, H] in body frame.
        """
        batch_size, vertical_num, horizon_num = (
            endstate_pred.shape[0],
            endstate_pred.shape[2],
            endstate_pred.shape[3],
        )
        device = endstate_pred.device

        endstate_pred = endstate_pred.permute(0, 2, 3, 1).reshape(batch_size, vertical_num * horizon_num, 9)

        yaw, pitch = self.lattice_primitive.get_angle_lattice(device=device)
        yaw = yaw.flip(0)[None, :].expand(batch_size, -1)
        pitch = pitch.flip(0)[None, :].expand(batch_size, -1)
        rotation_bp = self.lattice_primitive.get_rotation(device=device).flip(0)
        rotation_bp = rotation_bp[None, :, :, :].expand(batch_size, -1, -1, -1)

        delta_yaw = endstate_pred[:, :, 0] * float(self.lattice_primitive.yaw_diff)
        delta_pitch = endstate_pred[:, :, 1] * float(self.lattice_primitive.pitch_diff)
        radio = (endstate_pred[:, :, 2] + 1.0) * self.lattice_primitive.radio_range

        cos_pitch = torch.cos(pitch + delta_pitch)
        endstate_x = cos_pitch * torch.cos(yaw + delta_yaw) * radio
        endstate_y = cos_pitch * torch.sin(yaw + delta_yaw) * radio
        endstate_z = torch.sin(pitch + delta_pitch) * radio
        endstate_p = torch.stack([endstate_x, endstate_y, endstate_z], dim=-1)

        endstate_vp = endstate_pred[:, :, 3:6] * self.lattice_primitive.vel_max
        endstate_ap = endstate_pred[:, :, 6:9] * self.lattice_primitive.acc_max

        endstate_vb = torch.matmul(rotation_bp, endstate_vp.unsqueeze(-1)).squeeze(-1)
        endstate_ab = torch.matmul(rotation_bp, endstate_ap.unsqueeze(-1)).squeeze(-1)
        endstate = torch.cat([endstate_p, endstate_vb, endstate_ab], dim=-1)
        return endstate.permute(0, 2, 1).reshape(batch_size, 9, vertical_num, horizon_num)

    def pred_to_endstate_cpu(self, endstate_pred: np.ndarray, lattice_id) -> np.ndarray:
        """
        Numpy version of ``pred_to_endstate`` for deployment/test-time post-processing.

        Args:
            endstate_pred: [N, 9]
            lattice_id: scalar or [N]
        Returns:
            [N, 9] in body frame.
        """
        lattice_id_tensor = torch.as_tensor(lattice_id, dtype=torch.long)
        delta_yaw = endstate_pred[:, 0] * float(self.lattice_primitive.yaw_diff)
        delta_pitch = endstate_pred[:, 1] * float(self.lattice_primitive.pitch_diff)
        radio = (endstate_pred[:, 2] + 1.0) * self.lattice_primitive.radio_range

        yaw, pitch = self.lattice_primitive.get_angle_lattice(lattice_id_tensor)
        yaw = yaw.cpu().numpy()
        pitch = pitch.cpu().numpy()
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.stack((endstate_x, endstate_y, endstate_z), axis=1)

        endstate_vp = endstate_pred[:, 3:6] * self.lattice_primitive.vel_max
        endstate_ap = endstate_pred[:, 6:9] * self.lattice_primitive.acc_max

        rotation_pb = self.lattice_primitive.get_rotation(lattice_id_tensor).cpu().numpy()
        endstate_vb = np.matmul(rotation_pb, endstate_vp[:, :, np.newaxis]).squeeze(-1)
        endstate_ab = np.matmul(rotation_pb, endstate_ap[:, :, np.newaxis]).squeeze(-1)
        return np.concatenate((endstate_p, endstate_vb, endstate_ab), axis=1)

    def prepare_input(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Transform normalized body-frame observations to primitive-frame feature maps.

        Args:
            obs: [B, 9]
        Returns:
            [B, 9, V, H]
        """
        batch_size, traj_num = obs.shape[0], self.lattice_primitive.traj_num
        rotation_bp_all = self.lattice_primitive.get_rotation(device=obs.device).flip(0)

        obs = obs.view(batch_size, 3, 3)
        obs_expanded = obs[:, None, :, :].expand(batch_size, traj_num, 3, 3)
        rotation_bp = rotation_bp_all[None, :, :, :].expand(batch_size, traj_num, 3, 3)

        transformed = torch.matmul(obs_expanded, rotation_bp)
        transformed = transformed.view(batch_size, traj_num, 9)
        transformed = transformed.permute(0, 2, 1).contiguous()
        return transformed.view(
            batch_size,
            9,
            self.lattice_primitive.vertical_num,
            self.lattice_primitive.horizon_num,
        )

    def unnormalize_obs(self, vel_acc: torch.Tensor) -> torch.Tensor:
        out = vel_acc.clone()
        out[:, 0:3] = out[:, 0:3] * self.lattice_primitive.vel_max
        out[:, 3:6] = out[:, 3:6] * self.lattice_primitive.acc_max
        return out

    def normalize_obs(self, vel_acc_goal: torch.Tensor) -> torch.Tensor:
        out = vel_acc_goal.clone()
        out[:, 0:3] = out[:, 0:3] / self.lattice_primitive.vel_max
        out[:, 3:6] = out[:, 3:6] / self.lattice_primitive.acc_max

        goal_norm = out[:, 6:9].norm(dim=1, keepdim=True)
        out[:, 6:9] = out[:, 6:9] / goal_norm.clamp(min=self.goal_length)
        return out


def rotate_body2world(rot_wb: torch.Tensor, pos_b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)


def transform_body2world(rot_wb: torch.Tensor, t_w: torch.Tensor, pos_b: torch.Tensor) -> torch.Tensor:
    return rotate_body2world(rot_wb, pos_b) + t_w


def state_body2world(
    pos_w: torch.Tensor,
    rot_wb: torch.Tensor,
    pos_b: torch.Tensor,
    vel_b: torch.Tensor,
    acc_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_b = transform_body2world(rot_wb, pos_w, pos_b)
    vel_b = rotate_body2world(rot_wb, vel_b)
    acc_b = rotate_body2world(rot_wb, acc_b)
    return pos_b, vel_b, acc_b
