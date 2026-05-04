import importlib.util
from pathlib import Path

from pls.sensors.registry import build_sensor_model
from pls.workflows.registry import resolve_runtime_bundle


def _load_research_registration_module():
    path = Path('examples/train_a_policy/frozenlake_plugin/research_registration.py').resolve()
    spec = importlib.util.spec_from_file_location('research_registration', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_research_registration_and_sensor():
    module = _load_research_registration_module()
    module.register_for_research()

    runtime = resolve_runtime_bundle('FrozenLake-v1', 'a2c')
    assert runtime.model_cls is not None

    sensor = build_sensor_model('frozenlake_grid_sensor_v1', grid_size=4)
    out = sensor.predict(0)
    assert out.shape == (1, 4)
    # state 0 (start): no move leads directly to a hole
    assert out.tolist()[0] == [0.0, 0.0, 0.0, 0.0]

    # state 6 (row=1,col=2): left->5 hole, right->7 hole
    out6 = sensor.predict(6)
    assert out6.tolist()[0] == [1.0, 0.0, 1.0, 0.0]
