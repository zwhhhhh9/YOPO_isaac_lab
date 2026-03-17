"""Test-time YOPO model with lightweight post-processing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from yopo_drone.network.models.common import Config, YopoNetworkCore, make_config


class YopoTestModel(YopoNetworkCore):
    """YOPO deployment model with CPU-side decoding helpers."""

    def __init__(
        self,
        *,
        observation_dim: int = 9,
        output_dim: int = 10,
        hidden_state: int = 64,
        compact_backbone: bool = False,
        config: Config | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            observation_dim=observation_dim,
            output_dim=output_dim,
            hidden_state=hidden_state,
            compact_backbone=compact_backbone,
            config=make_config(config, train=False),
            device=device,
        )
        self.lattice_primitive = self.state_transform.lattice_primitive

    @torch.inference_mode()
    def predict(
        self,
        depth: torch.Tensor,
        obs_body: torch.Tensor,
        *,
        return_all_predictions: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = self.prepare_obs(obs_body)
        endstate_pred, score_pred = self.forward(depth, obs)
        return self.postprocess(endstate_pred, score_pred, return_all_predictions=return_all_predictions)

    def postprocess(
        self,
        endstate_pred: torch.Tensor | np.ndarray,
        score_pred: torch.Tensor | np.ndarray,
        *,
        return_all_predictions: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        endstate_np = self._to_numpy(endstate_pred)
        score_np = self._to_numpy(score_pred)

        batch_size = endstate_np.shape[0]
        traj_num = self.lattice_primitive.traj_num
        endstate_flat = endstate_np.reshape(batch_size, 9, traj_num).transpose(0, 2, 1)
        score_flat = score_np.reshape(batch_size, traj_num)

        if return_all_predictions:
            lattice_ids = torch.arange(traj_num - 1, -1, -1, dtype=torch.long)
            decoded = np.stack(
                [
                    self.state_transform.pred_to_endstate_cpu(endstate_flat[batch_index], lattice_ids)
                    for batch_index in range(batch_size)
                ],
                axis=0,
            )
            action_ids = np.tile(np.arange(traj_num, dtype=np.int64), (batch_size, 1))
            return decoded, score_flat, action_ids

        action_ids = np.argmin(score_flat, axis=1).astype(np.int64)
        best_endstates = []
        best_scores = []
        for batch_index, action_id in enumerate(action_ids):
            lattice_id = self.lattice_primitive.convert_image_grid_lattice_id(int(action_id))
            decoded = self.state_transform.pred_to_endstate_cpu(
                endstate_flat[batch_index, action_id : action_id + 1],
                lattice_id,
            )
            best_endstates.append(decoded[0])
            best_scores.append(score_flat[batch_index, action_id])

        return np.stack(best_endstates, axis=0), np.asarray(best_scores, dtype=np.float32), action_ids

    @staticmethod
    def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def prepare_depth_tensor(depth: Any, *, device: torch.device | str | None = None) -> torch.Tensor:
        if isinstance(depth, torch.Tensor):
            tensor = depth.float()
        else:
            tensor = torch.as_tensor(depth, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected depth input with 3 or 4 dims, got shape {tuple(tensor.shape)}")
        return tensor.to(device=device) if device is not None else tensor

