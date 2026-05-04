from typing import Any, Optional
import torch as th
from pls.sensors.base import SensorModel


class MySensorModel(SensorModel):
    def __init__(self, num_sensors: int):
        self.num_sensors = num_sensors

    def predict(self, obs: Any, info: Optional[dict] = None) -> th.Tensor:
        x = th.as_tensor(obs, dtype=th.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        out = th.zeros((x.shape[0], self.num_sensors), dtype=th.float32)
        return out
