from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th

from pls.sensors.base import SensorModel


class RuleBasedSensorModel(SensorModel):
    """Deterministic sensor model that maps normalized line position to sensors.

    Sensor order:
    0: near_left_edge
    1: near_right_edge
    2: at_center
    """

    def __init__(self, length: int, include_center: bool = True):
        self.length = int(length)
        self.include_center = include_center
        self.center_pos = self.length // 2

    def _to_tensor(self, obs: Any) -> th.Tensor:
        if isinstance(obs, th.Tensor):
            out = obs.detach().float()
        else:
            out = th.as_tensor(np.asarray(obs), dtype=th.float32)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        return out

    def predict(self, obs: Any, info: Optional[dict] = None) -> th.Tensor:
        x = self._to_tensor(obs)
        if x.size(-1) != 1:
            raise ValueError(f"Expected obs with last dim=1, got {tuple(x.shape)}")

        pos = th.round(x[:, 0] * float(self.length - 1)).long()
        near_left = (pos <= 1).float()
        near_right = (pos >= self.length - 2).float()

        sensors = [near_left, near_right]
        if self.include_center:
            at_center = (pos == self.center_pos).float()
            sensors.append(at_center)

        return th.stack(sensors, dim=1)


def build_rule_based_lineworld_sensor(**kwargs) -> RuleBasedSensorModel:
    length = kwargs.get("length", 7)
    include_center = kwargs.get("include_center", True)
    return RuleBasedSensorModel(length=length, include_center=include_center)
