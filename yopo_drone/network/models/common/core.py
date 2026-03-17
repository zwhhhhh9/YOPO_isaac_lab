"""Shared YOPO network core."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .backbone import YopoBackbone
from .config import Config, make_config
from .head import YopoHead
from .state_transform import StateTransform


class YopoNetworkCore(nn.Module):
    """
    Shared YOPO architecture.

    ``forward`` expects primitive-frame observations with shape ``[B, 9, V, H]``.
    Use ``prepare_obs`` / ``inference`` when the observation is still in body frame.
    """

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
        super().__init__()
        self.cfg = make_config(config)
        self.state_transform = StateTransform(self.cfg)

        self.image_backbone = YopoBackbone(hidden_state, compact=compact_backbone)
        self.state_backbone = nn.Identity()
        self.yopo_head = YopoHead(hidden_state + observation_dim, output_dim)
        if device is not None:
            self.to(device)

    def forward(self, depth: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth_feature = self.image_backbone(depth)
        obs_feature = self.state_backbone(obs)
        input_tensor = torch.cat((obs_feature, depth_feature), dim=1)
        output = self.yopo_head(input_tensor)
        endstate = torch.tanh(output[:, :9])
        score = torch.nn.functional.softplus(output[:, 9])
        return endstate, score

    def prepare_obs(self, obs_body: torch.Tensor) -> torch.Tensor:
        obs_normalized = self.state_transform.normalize_obs(obs_body)
        return self.state_transform.prepare_input(obs_normalized)

    def decode_predictions(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        return self.state_transform.pred_to_endstate(endstate_pred)

    def inference(self, depth: torch.Tensor, obs_body: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.prepare_obs(obs_body)
        endstate_pred, score_pred = self.forward(depth, obs)
        return self.decode_predictions(endstate_pred), score_pred

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> None:
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict, strict=strict)
