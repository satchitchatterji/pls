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
    """Simple state-based sensor wrapper for FrozenLake discrete states.

    Sensor order:
    0: at_start
    1: at_goal
    2: in_top_row
    3: in_left_col
    """

    def __init__(self, grid_size: int = 4):
        self.grid_size = int(grid_size)
        self.n_states = self.grid_size * self.grid_size

    def predict(self, obs, info=None):
        del info
        if isinstance(obs, th.Tensor):
            s = obs.detach().long().reshape(-1)
        else:
            s = th.as_tensor(np.asarray(obs), dtype=th.long).reshape(-1)

        at_start = (s == 0).float()
        at_goal = (s == (self.n_states - 1)).float()
        row = th.div(s, self.grid_size, rounding_mode="floor")
        col = s % self.grid_size
        in_top_row = (row == 0).float()
        in_left_col = (col == 0).float()
        return th.stack([at_start, at_goal, in_top_row, in_left_col], dim=1)


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
