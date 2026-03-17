"""Training-time YOPO model."""

from __future__ import annotations

import torch

from yopo_drone.network.models.common import Config, YopoNetworkCore, make_config


class YopoTrainModel(YopoNetworkCore):
    """YOPO model used during supervised trajectory training."""

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
            config=make_config(config, train=True),
            device=device,
        )

