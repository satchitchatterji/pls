from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch as th


class SensorModel(ABC):
    """Abstract sensor model producing probabilistic sensor values."""

    @abstractmethod
    def predict(self, obs: Any, info: Optional[dict] = None) -> th.Tensor:
        """Return batched sensor values in [0, 1] with shape [batch, num_sensors]."""
        raise NotImplementedError
