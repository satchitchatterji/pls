import tempfile
from pathlib import Path

import pytest
import torch as th

from pls.algorithms.learn import _validate_preflight
from pls.envs.line_world import LineWorldSafetyEnv
from pls.sensors.base import SensorModel
from pls.workflows.registry import RuntimeBundle, register_env_runtime, resolve_runtime_bundle


class DummySensor(SensorModel):
    def __init__(self, width: int):
        self.width = width

    def predict(self, obs, info=None):
        x = th.as_tensor(obs, dtype=th.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return th.zeros((x.shape[0], self.width), dtype=th.float32)


def test_registry_custom_env_runtime():
    env_id = "UnitTestEnv-v0"

    def resolver(algo_cls):
        return RuntimeBundle(model_cls=algo_cls, get_sensor_value_ground_truth=lambda x: x)

    register_env_runtime(env_id, resolver)
    runtime = resolve_runtime_bundle(env_id, "a2c")
    assert runtime.model_cls is not None


def test_preflight_raises_on_action_mismatch():
    env = LineWorldSafetyEnv(length=7, max_steps=10)
    shield_params = {
        "num_sensors": 3,
        "num_actions": 5,
        "shield_program": "dummy",
    }
    with pytest.raises(ValueError, match="num_actions"):
        _validate_preflight(env, None, shield_params, {})


def test_preflight_raises_on_sensor_width_mismatch():
    env = LineWorldSafetyEnv(length=7, max_steps=10)
    sensor = DummySensor(width=2)
    shield_params = {
        "num_sensors": 3,
        "num_actions": 3,
        "shield_program": "dummy",
    }
    with pytest.raises(ValueError, match="sensor output width"):
        _validate_preflight(env, sensor, shield_params, {})
