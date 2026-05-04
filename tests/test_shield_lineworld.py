import pathlib

import torch as th

from pls.shields.shields import Shield
from pls.sensors.rule_based import RuleBasedSensorModel


def _program_path() -> str:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    return str(repo_root / "examples" / "train_a_policy" / "data" / "lineworld_safety.pl")


def test_shield_reduces_unsafe_action_probability():
    sensor = RuleBasedSensorModel(length=7, include_center=True)
    shield = Shield(
        config_folder=str(pathlib.Path(_program_path()).parent),
        shield_program=pathlib.Path(_program_path()).name,
        num_sensors=3,
        num_actions=3,
        differentiable=True,
        sensor_model=sensor,
    )

    # near left edge => left action should be reduced to zero-ish
    obs = th.tensor([[1.0 / 6.0]], dtype=th.float32)
    sensor_values = shield.get_sensor_values(obs)
    base_actions = th.tensor([[0.8, 0.1, 0.1]], dtype=th.float32)

    shielded = shield.get_shielded_policy(base_actions, sensor_values)

    assert shielded.shape == base_actions.shape
    assert th.allclose(shielded.sum(dim=1), th.ones(1), atol=1e-4)
    assert shielded[0, 0].item() < 1e-4
