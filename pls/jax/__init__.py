"""Optional JAX backend for CleanPLS research experiments.

The existing Torch/SB3 backend remains the default.  This namespace contains
small, dependency-isolated JAX equivalents used by the FrozenLake examples.
"""

from .frozenlake import (
    FrozenLakeSensorConfig,
    FrozenLakeJaxSensorModel,
    FrozenLakeMLPSensorModel,
    build_supervised_dataset,
    load_sensor,
    pretrain_sensor,
    predict_sensor,
    save_sensor,
    shield_policy,
)
from .shields import FrozenLakeJaxShield, FrozenLakeShieldConfig
from .pipeline import FrozenLakeJaxPipelineConfig, run_pipeline_benchmark

__all__ = [
    "FrozenLakeSensorConfig",
    "FrozenLakeJaxSensorModel",
    "FrozenLakeMLPSensorModel",
    "build_supervised_dataset",
    "load_sensor",
    "pretrain_sensor",
    "predict_sensor",
    "save_sensor",
    "shield_policy",
    "FrozenLakeJaxShield",
    "FrozenLakeShieldConfig",
    "FrozenLakeJaxPipelineConfig",
    "run_pipeline_benchmark",
]
