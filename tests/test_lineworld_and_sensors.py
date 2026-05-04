import numpy as np
import torch as th

from pls.envs.line_world import LineWorldSafetyEnv
from pls.sensors.rule_based import RuleBasedSensorModel


def test_lineworld_reset_and_step():
    env = LineWorldSafetyEnv(length=7, max_steps=10)
    obs, info = env.reset(seed=123)

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert info["position"] == 3

    obs, reward, terminated, truncated, info = env.step(1)
    assert reward == 0.1
    assert not terminated
    assert not truncated
    assert info["position"] == 3


def test_lineworld_unsafe_boundary():
    env = LineWorldSafetyEnv(length=7, max_steps=10)
    env.reset(seed=0)

    # Move left until unsafe terminal
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == -1.0
    assert info["unsafe"] is True
    assert info["position"] == 0


def test_rule_based_sensor_outputs():
    sensor = RuleBasedSensorModel(length=7, include_center=True)

    left = sensor.predict(np.array([1.0 / 6.0], dtype=np.float32))
    center = sensor.predict(np.array([3.0 / 6.0], dtype=np.float32))
    right = sensor.predict(np.array([5.0 / 6.0], dtype=np.float32))

    assert th.equal(left, th.tensor([[1.0, 0.0, 0.0]]))
    assert th.equal(center, th.tensor([[0.0, 0.0, 1.0]]))
    assert th.equal(right, th.tensor([[0.0, 1.0, 0.0]]))
    assert left.dtype == th.float32
