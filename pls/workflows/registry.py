from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from pls.algorithms.a2c_shielded import A2C_shielded
from pls.algorithms.dqn import DQN_standard
from pls.algorithms.ddpg_shielded import DDPG_shielded
from pls.algorithms.double_dqn_shielded import DoubleDQN_shielded
from pls.algorithms.dqn_shielded import DQN_shielded
from pls.algorithms.ppo_shielded import PPO_shielded
from pls.algorithms.rainbow_shielded import Rainbow_shielded
from pls.algorithms.sac_shielded import SAC_shielded
from pls.algorithms.td3_shielded import TD3_shielded
from pls.algorithms.trpo_shielded import TRPO_shielded


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
    "double_dqn": DoubleDQN_shielded,
    "rainbow": Rainbow_shielded,
    "sac": SAC_shielded,
    "td3": TD3_shielded,
    "ddpg": DDPG_shielded,
    "trpo": TRPO_shielded,
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


register_env_runtime("LineWorldSafety-v0", _lineworld_resolver)


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
