"""JAX implementation of the FrozenLake sensor and discrete shield.

This is intentionally a parallel backend rather than a silent replacement for
the Torch/SB3 implementation.  It provides the same sensor semantics:

* one-hot FrozenLake state input,
* four action-to-hole risk outputs in ``[left, down, right, up]`` order,
* BCE-with-logits pretraining,
* differentiable action redistribution through a safety mask.

JAX selects the execution device through its installed backend.  On Apple
Silicon, CPU execution uses standard JAX and GPU execution uses ``jax-metal``
when the Metal plugin is installed and compatible with the local JAX version.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from pls.sensors.base import SensorModel


@dataclass(frozen=True)
class FrozenLakeSensorConfig:
    """Architecture and dataset settings for the FrozenLake sensor."""

    grid_size: int = 4
    hidden_sizes: tuple[int, ...] = (16, 16)
    out_dim: int = 4


class FrozenLakeMLPSensorModel(SensorModel):
    """Non-trainable JAX sensor wrapper using the CleanPLS sensor contract.

    The wrapper stores only inference parameters.  Training stays explicit in
    :func:`pretrain_sensor`, which makes it easy to separate sensor-data
    generation, pretraining, and downstream shielded RL experiments.
    """

    def __init__(self, params, config: FrozenLakeSensorConfig = FrozenLakeSensorConfig()):
        self.params = params
        self.config = config

    @classmethod
    def from_checkpoint(cls, path: str | Path):
        """Load a frozen sensor from a ``save_sensor`` NumPy archive."""
        params, config = load_sensor(path)
        return cls(params, config)

    def predict(self, obs, info=None) -> th.Tensor:
        """Return per-action hole probabilities as a CPU Torch tensor."""
        del info
        return th.as_tensor(
            np.asarray(predict_sensor(self.params, obs, self.config)).copy(),
            dtype=th.float32,
        )


class FrozenLakeJaxSensorModel:
    """Pure-JAX sensor wrapper that keeps predictions on the JAX device."""

    def __init__(self, params, config: FrozenLakeSensorConfig = FrozenLakeSensorConfig()):
        self.params = params
        self.config = config

    @classmethod
    def from_checkpoint(cls, path: str | Path):
        """Load a JAX sensor checkpoint without converting it to Torch."""
        params, config = load_sensor(path)
        return cls(params, config)

    def predict(self, obs, info=None):
        """Return JAX-array hole probabilities on the active JAX device."""
        del info
        return predict_sensor(self.params, obs, self.config)


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The JAX backend requires the optional dependency. "
            "Install with `pip install -e '.[jax]'`."
        ) from exc
    return jax, jnp


def next_state(state: int, action: int, grid_size: int = 4) -> int:
    """Return intended deterministic grid transition with boundary clamping."""
    row, col = divmod(int(state), grid_size)
    if action == 0:
        col2, row2 = max(0, col - 1), row
    elif action == 1:
        col2, row2 = col, min(grid_size - 1, row + 1)
    elif action == 2:
        col2, row2 = min(grid_size - 1, col + 1), row
    elif action == 3:
        col2, row2 = col, max(0, row - 1)
    else:
        raise ValueError(f"Unknown FrozenLake action: {action}")
    return row2 * grid_size + col2


def build_supervised_dataset(
    config: FrozenLakeSensorConfig = FrozenLakeSensorConfig(),
    hole_states: tuple[int, ...] = (5, 7, 11, 12),
) -> tuple[np.ndarray, np.ndarray]:
    """Build one-hot states and action-to-hole labels.

    Returns ``x`` with shape ``[16, 16]`` and ``y`` with shape ``[16, 4]`` for
    the default 4x4 map.
    """
    n_states = config.grid_size * config.grid_size
    states = np.arange(n_states, dtype=np.int32)
    x = np.eye(n_states, dtype=np.float32)
    holes = set(hole_states)
    y = np.asarray(
        [
            [
                float(next_state(int(s), action, config.grid_size) in holes)
                for action in range(config.out_dim)
            ]
            for s in states
        ],
        dtype=np.float32,
    )
    return x, y


def _init_params(key, config: FrozenLakeSensorConfig):
    jax, jnp = _require_jax()
    dims = (config.grid_size * config.grid_size, *config.hidden_sizes, config.out_dim)
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for subkey, in_dim, out_dim in zip(keys, dims[:-1], dims[1:]):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        weight = jax.random.uniform(
            subkey, (in_dim, out_dim), minval=-limit, maxval=limit
        )
        params.append({"w": weight, "b": jnp.zeros((out_dim,), dtype=jnp.float32)})
    return params


def _apply(params, x):
    _, jnp = _require_jax()
    hidden = x
    for layer in params[:-1]:
        hidden = jnp.maximum(hidden @ layer["w"] + layer["b"], 0.0)
    last = params[-1]
    return hidden @ last["w"] + last["b"]


def _bce_with_logits(logits, labels):
    _, jnp = _require_jax()
    return jnp.mean(jnp.maximum(logits, 0.0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def _adam_init(params):
    _, jnp = _require_jax()
    return {
        "m": [{"w": jnp.zeros_like(p["w"]), "b": jnp.zeros_like(p["b"])} for p in params],
        "v": [{"w": jnp.zeros_like(p["w"]), "b": jnp.zeros_like(p["b"])} for p in params],
        "step": 0,
    }


def _adam_update(params, grads, state, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
    _, jnp = _require_jax()
    step = state["step"] + 1
    new_params = []
    new_m = []
    new_v = []
    for p, g, m, v in zip(params, grads, state["m"], state["v"]):
        layer_params = {}
        layer_m = {}
        layer_v = {}
        for name in ("w", "b"):
            m_new = beta1 * m[name] + (1 - beta1) * g[name]
            v_new = beta2 * v[name] + (1 - beta2) * jnp.square(g[name])
            m_hat = m_new / (1 - beta1**step)
            v_hat = v_new / (1 - beta2**step)
            layer_params[name] = p[name] - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
            layer_m[name] = m_new
            layer_v[name] = v_new
        new_params.append(layer_params)
        new_m.append(layer_m)
        new_v.append(layer_v)
    return new_params, {"m": new_m, "v": new_v, "step": step}


def pretrain_sensor(
    *,
    seed: int = 0,
    epochs: int = 120,
    learning_rate: float = 1e-2,
    config: FrozenLakeSensorConfig = FrozenLakeSensorConfig(),
    device: Any = None,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Pretrain the JAX sensor and return parameters plus BCE history."""
    jax, jnp = _require_jax()
    x, y = build_supervised_dataset(config)
    key = jax.random.PRNGKey(seed)
    params = _init_params(key, config)
    state = _adam_init(params)

    x_device = jnp.asarray(x)
    y_device = jnp.asarray(y)
    if device is not None:
        # Trigger placement before timing/training so callers can select CPU or Metal.
        with jax.default_device(device):
            params = jax.tree_util.tree_map(jax.device_put, params)
            state = jax.tree_util.tree_map(jax.device_put, state)
            x_device = jax.device_put(x_device, device)
            y_device = jax.device_put(y_device, device)

    def loss_fn(current_params):
        return _bce_with_logits(_apply(current_params, x_device), y_device)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    losses: list[float] = []
    for _ in range(epochs):
        loss, grads = value_and_grad(params)
        params, state = _adam_update(params, grads, state, learning_rate)
        losses.append(float(loss))
    jax.block_until_ready(params)
    return params, losses


def predict_sensor(params, observations, config: FrozenLakeSensorConfig = FrozenLakeSensorConfig()):
    """Return sigmoid sensor probabilities for scalar or batched state ids."""
    _, jnp = _require_jax()
    states = jnp.asarray(observations, dtype=jnp.int32).reshape(-1)
    features = jnp.eye(config.grid_size * config.grid_size, dtype=jnp.float32)[states]
    return jax_sigmoid(_apply(params, features))


def jax_sigmoid(x):
    """Small import-safe sigmoid helper."""
    _, jnp = _require_jax()
    return jnp.reciprocal(1.0 + jnp.exp(-x))


def shield_policy(base_probs, sensor_values, eps: float = 1e-8):
    """Redistribute a categorical policy over actions predicted safe.

    ``sensor_values`` contains per-action hole probabilities.  The safety mask
    is ``1 - sensor_values`` and the result is normalized per batch item.
    """
    _, jnp = _require_jax()
    base_probs = jnp.asarray(base_probs)
    safety = jnp.clip(1.0 - jnp.asarray(sensor_values), 0.0, 1.0)
    masked = base_probs * safety
    denominator = jnp.sum(masked, axis=-1, keepdims=True)
    fallback = jnp.full_like(masked, 1.0 / masked.shape[-1])
    return jnp.where(denominator > eps, masked / (denominator + eps), fallback)


def save_sensor(path: str | Path, params, config: FrozenLakeSensorConfig = FrozenLakeSensorConfig()):
    """Save JAX sensor parameters as a portable NumPy archive."""
    _, jnp = _require_jax()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for index, layer in enumerate(params):
        flat[f"layer_{index}_w"] = np.asarray(jax_block_until_ready(jnp, layer["w"]))
        flat[f"layer_{index}_b"] = np.asarray(jax_block_until_ready(jnp, layer["b"]))
    flat["grid_size"] = np.asarray(config.grid_size)
    flat["hidden_sizes"] = np.asarray(config.hidden_sizes, dtype=np.int32)
    flat["out_dim"] = np.asarray(config.out_dim)
    np.savez(path, **flat)


def load_sensor(path: str | Path):
    """Load a parameter archive produced by :func:`save_sensor`."""
    jax, jnp = _require_jax()
    data = np.load(path)
    hidden_sizes = tuple(int(x) for x in data["hidden_sizes"])
    config = FrozenLakeSensorConfig(
        grid_size=int(data["grid_size"]), hidden_sizes=hidden_sizes, out_dim=int(data["out_dim"])
    )
    params = []
    for index in range(len(hidden_sizes) + 1):
        params.append(
            {
                "w": jnp.asarray(data[f"layer_{index}_w"]),
                "b": jnp.asarray(data[f"layer_{index}_b"]),
            }
        )
    return params, config


def jax_block_until_ready(jnp, value):
    """Block on an array without requiring callers to import JAX internals."""
    return value.block_until_ready() if hasattr(value, "block_until_ready") else value
