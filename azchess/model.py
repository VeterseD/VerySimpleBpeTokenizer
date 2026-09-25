"""AlphaZero ResNet: shared residual tower with policy and value heads."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .encoding import HALFMOVE_PLANE, NUM_PLANES, POLICY_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AlphaZeroNet(nn.Module):
    def __init__(self, blocks: int = 10, channels: int = 128):
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels}
        scale = torch.ones(NUM_PLANES)
        scale[HALFMOVE_PLANE] = 1 / 100
        self.register_buffer("input_scale", scale.view(1, -1, 1, 1), persistent=False)

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, POLICY_PLANES, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: uint8/float planes (B, NUM_PLANES, 8, 8) -> policy logits (B, 4672), value (B,) in [-1, 1]."""
        x = x.float() * self.input_scale
        h = self.tower(self.stem(x))
        return self.policy_head(h).flatten(1), self.value_head(h).squeeze(1)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype_for(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def save_checkpoint(path: Path, model: AlphaZeroNet, optimizer=None, iteration: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "iteration": iteration,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[AlphaZeroNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = AlphaZeroNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
