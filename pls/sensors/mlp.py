from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch as th
import torch.nn as nn

from pls.sensors.base import SensorModel


class LineWorldSensorMLP(nn.Module):
    def __init__(self, hidden_sizes=(32, 32), out_dim: int = 3):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


@dataclass
class LineWorldMLPSensorConfig:
    checkpoint_path: str
    hidden_sizes: tuple[int, ...] = (32, 32)
    include_center: bool = True
    threshold: float = 0.5


class PretrainedLineWorldMLPSensorModel(SensorModel):
    """Pretrained MLP sensor model for LineWorld.

    Sensor order:
    0: near_left_edge
    1: near_right_edge
    2: at_center (if include_center=True)
    """

    def __init__(self, checkpoint_path: str, hidden_sizes=(32, 32), include_center=True, threshold=0.5):
        self.include_center = include_center
        self.threshold = float(threshold)
        out_dim = 3 if include_center else 2
        self.model = LineWorldSensorMLP(hidden_sizes=hidden_sizes, out_dim=out_dim)
        state = th.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            self.model.load_state_dict(state["state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()

    def _to_tensor(self, obs: Any) -> th.Tensor:
        if isinstance(obs, th.Tensor):
            x = obs.detach().float()
        else:
            x = th.as_tensor(np.asarray(obs), dtype=th.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.size(-1) != 1:
            raise ValueError(f"Expected obs with last dim=1, got {tuple(x.shape)}")
        return x

    def predict(self, obs: Any, info: Optional[dict] = None) -> th.Tensor:
        del info
        x = self._to_tensor(obs)
        with th.no_grad():
            probs = th.sigmoid(self.model(x))
            # Keep probabilistic output contract, deterministic thresholding can be done downstream if needed.
            return probs.clamp(0.0, 1.0)


def build_pretrained_lineworld_mlp_sensor(**kwargs) -> PretrainedLineWorldMLPSensorModel:
    checkpoint_path = kwargs.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for pretrained_lineworld_mlp_v1")
    hidden_sizes = tuple(kwargs.get("hidden_sizes", [32, 32]))
    include_center = kwargs.get("include_center", True)
    threshold = kwargs.get("threshold", 0.5)
    return PretrainedLineWorldMLPSensorModel(
        checkpoint_path=checkpoint_path,
        hidden_sizes=hidden_sizes,
        include_center=include_center,
        threshold=threshold,
    )
