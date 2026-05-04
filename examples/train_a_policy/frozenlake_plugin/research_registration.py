"""Research-side FrozenLake integration example.

This file is intentionally outside `pls/` to demonstrate the workflow where
researchers import CleanPLS and register their own env runtime + sensors with
minimal framework edits.
"""

from __future__ import annotations

from typing import Type

import gymnasium as gym

from pls.sensors.base import SensorModel
from pls.sensors.registry import SENSOR_REGISTRY
from pls.workflows.registry import RuntimeBundle, register_env_runtime

import numpy as np
import torch as th


class FrozenLakeGridSensor(SensorModel):
    """State-based safety sensor wrapper for FrozenLake discrete states.

    Sensor order:
    0: left_to_hole
    1: down_to_hole
    2: right_to_hole
    3: up_to_hole
    """

    def __init__(self, grid_size: int = 4):
        self.grid_size = int(grid_size)
        self.n_states = self.grid_size * self.grid_size
        # Default FrozenLake-4x4 layout:
        # S F F F
        # F H F H
        # F F F H
        # H F F G
        # Hole states: 5, 7, 11, 12
        self.holes = {5, 7, 11, 12}

    def _next_state(self, s: th.Tensor, action: int) -> th.Tensor:
        row = th.div(s, self.grid_size, rounding_mode="floor")
        col = s % self.grid_size
        if action == 0:  # left
            col2 = th.clamp(col - 1, min=0)
            row2 = row
        elif action == 1:  # down
            row2 = th.clamp(row + 1, max=self.grid_size - 1)
            col2 = col
        elif action == 2:  # right
            col2 = th.clamp(col + 1, max=self.grid_size - 1)
            row2 = row
        elif action == 3:  # up
            row2 = th.clamp(row - 1, min=0)
            col2 = col
        else:
            raise ValueError(f"Unknown action: {action}")
        return row2 * self.grid_size + col2

    def predict(self, obs, info=None):
        del info
        if isinstance(obs, th.Tensor):
            s = obs.detach().long().reshape(-1)
        else:
            s = th.as_tensor(np.asarray(obs), dtype=th.long).reshape(-1)

        hole_ids = th.as_tensor(sorted(self.holes), dtype=th.long, device=s.device)

        def leads_to_hole(action: int) -> th.Tensor:
            ns = self._next_state(s, action)
            return (ns.unsqueeze(1) == hole_ids.unsqueeze(0)).any(dim=1).float()

        left_to_hole = leads_to_hole(0)
        down_to_hole = leads_to_hole(1)
        right_to_hole = leads_to_hole(2)
        up_to_hole = leads_to_hole(3)
        return th.stack([left_to_hole, down_to_hole, right_to_hole, up_to_hole], dim=1)


def build_frozenlake_grid_sensor(**kwargs):
    grid_size = kwargs.get("grid_size", 4)
    return FrozenLakeGridSensor(grid_size=grid_size)


def _frozenlake_ground_truth(input):
    return build_frozenlake_grid_sensor(grid_size=4).predict(input)


def _resolver(algorithm_cls: Type) -> RuntimeBundle:
    return RuntimeBundle(
        model_cls=algorithm_cls,
        get_sensor_value_ground_truth=_frozenlake_ground_truth,
        custom_callback_class=None,
        monitor_cls=None,
        features_extractor_cls=None,
        observation_net_cls=None,
    )


def register_for_research() -> None:
    # sanity-check env exists in Gymnasium
    gym.spec("FrozenLake-v1")

    SENSOR_REGISTRY["frozenlake_grid_sensor_v1"] = build_frozenlake_grid_sensor
    register_env_runtime("FrozenLake-v1", _resolver)
