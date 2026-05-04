import tempfile

import torch as th

from pls.sensors.mlp import LineWorldSensorMLP, PretrainedLineWorldMLPSensorModel


def test_pretrained_mlp_sensor_predict_shape_and_range():
    model = LineWorldSensorMLP(hidden_sizes=(8,), out_dim=3)
    with tempfile.NamedTemporaryFile(suffix='.pt') as f:
        th.save({'state_dict': model.state_dict()}, f.name)
        sensor = PretrainedLineWorldMLPSensorModel(
            checkpoint_path=f.name,
            hidden_sizes=(8,),
            include_center=True,
        )
        obs = th.tensor([[0.0], [0.5], [1.0]], dtype=th.float32)
        out = sensor.predict(obs)

    assert out.shape == (3, 3)
    assert th.all(out >= 0.0)
    assert th.all(out <= 1.0)
