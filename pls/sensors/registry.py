from __future__ import annotations

from typing import Callable, Dict

from pls.sensors.rule_based import build_rule_based_lineworld_sensor

SENSOR_REGISTRY: Dict[str, Callable[..., object]] = {
    "rule_based_lineworld_v1": build_rule_based_lineworld_sensor,
}


def build_sensor_model(name: str, **kwargs):
    if name not in SENSOR_REGISTRY:
        raise KeyError(f"Unknown sensor model: {name}")
    return SENSOR_REGISTRY[name](**kwargs)
