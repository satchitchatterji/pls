from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Type

from pls.algorithms.a2c_shielded import A2C_shielded
from pls.algorithms.dqn import DQN_standard
from pls.algorithms.dqn_shielded import DQN_shielded
from pls.algorithms.ppo_shielded import PPO_shielded


@dataclass
class RuntimeBundle:
    model_cls: Type
    get_sensor_value_ground_truth: Callable
    custom_callback_class: Optional[Type] = None
    monitor_cls: Optional[Type] = None
    features_extractor_cls: Optional[Type] = None
    observation_net_cls: Optional[Type] = None


ALGO_REGISTRY: Dict[str, Type] = {
    "ppo": PPO_shielded,
    "a2c": A2C_shielded,
    "dqn_vanilla": DQN_standard,
    "dqn": DQN_shielded,
}


# Resolver receives the algorithm class and returns runtime bundle.
ENV_REGISTRY: Dict[str, Callable[[Type], RuntimeBundle]] = {}


def _noop_sensor(input):
    return input


def register_env_runtime(env_id: str, resolver: Callable[[Type], RuntimeBundle]) -> None:
    ENV_REGISTRY[env_id] = resolver


def _lineworld_resolver(algorithm_cls: Type) -> RuntimeBundle:
    return RuntimeBundle(
        model_cls=algorithm_cls,
        get_sensor_value_ground_truth=_noop_sensor,
    )


def _pacman_resolver(algorithm_cls: Type) -> RuntimeBundle:
    from env_specific_classes.pacman.env_classes import (
        Pacman_Callback,
        Pacman_FeaturesExtractor,
        Pacman_Monitor,
        Pacman_Observation_Net,
    )
    from env_specific_classes.pacman.util import get_ground_wall

    wall_color = 0.25
    ghost_color = 0.5
    pacman_color = 0.75
    ghost_distance = 1

    return RuntimeBundle(
        model_cls=algorithm_cls,
        get_sensor_value_ground_truth=partial(
            get_ground_wall, ghost_distance, pacman_color, ghost_color
        ),
        custom_callback_class=Pacman_Callback,
        monitor_cls=Pacman_Monitor,
        features_extractor_cls=Pacman_FeaturesExtractor,
        observation_net_cls=Pacman_Observation_Net,
    )


def _carracing_resolver(algorithm_cls: Type) -> RuntimeBundle:
    from env_specific_classes.carracing.env_classes import (
        Carracing_Callback,
        Carracing_FeaturesExtractor,
        Carracing_Monitor,
        Carracing_Observation_Net,
    )
    from env_specific_classes.carracing.util import get_ground_truth_of_grass

    return RuntimeBundle(
        model_cls=algorithm_cls,
        get_sensor_value_ground_truth=get_ground_truth_of_grass,
        custom_callback_class=Carracing_Callback,
        monitor_cls=Carracing_Monitor,
        features_extractor_cls=Carracing_FeaturesExtractor,
        observation_net_cls=Carracing_Observation_Net,
    )


register_env_runtime("LineWorldSafety-v0", _lineworld_resolver)
register_env_runtime("Pacman-v0", _pacman_resolver)
register_env_runtime("CarRacingPLS-v1", _carracing_resolver)


def resolve_runtime_bundle(env_name: str, algorithm: str) -> RuntimeBundle:
    algo_key = algorithm.lower()
    if algo_key not in ALGO_REGISTRY:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Supported: {sorted(ALGO_REGISTRY.keys())}"
        )

    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown env '{env_name}'. Register it via pls.workflows.registry.register_env_runtime(...)"
        )

    return ENV_REGISTRY[env_name](ALGO_REGISTRY[algo_key])
