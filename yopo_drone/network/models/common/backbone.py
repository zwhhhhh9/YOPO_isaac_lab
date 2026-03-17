"""YOPO visual backbones."""

from __future__ import annotations

import torch

from .resnet import resnet18


class ResNet18(torch.nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.cnn = resnet18(out_channels=output_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


class ResNet14(torch.nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.cnn = resnet18(out_channels=output_dim, truncate_last_stage=True)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


def YopoBackbone(output_dim: int, compact: bool = False) -> torch.nn.Module:
    return ResNet14(output_dim) if compact else ResNet18(output_dim)

