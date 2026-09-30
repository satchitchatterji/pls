"""JAX-native shields for discrete CleanPLS experiments.

The FrozenLake shield has one sensor per action.  A sensor value is the
probability that the action falls into a hole, so the corresponding safety
probability is ``1 - sensor_value``.  This is the JAX equivalent of the
FrozenLake ProbLog rules used by the Torch/SB3 examples:

``unsafe_next(Action) :- ActionToHole(Action).``
``safe_next(Action) :- Action(Action), not unsafe_next(Action).``

The implementation operates entirely on JAX arrays.  If its inputs and the
selected JAX device are Metal arrays, the masking and normalization execute on
the Apple GPU as part of the compiled JAX computation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .frozenlake import _require_jax, shield_policy


@dataclass(frozen=True)
class FrozenLakeShieldConfig:
    """Numerical settings for the four-action FrozenLake shield."""

    num_actions: int = 4
    eps: float = 1e-8


class FrozenLakeJaxShield:
    """JAX-native action shield for the FrozenLake hole-risk sensors.

    Parameters
    ----------
    config:
        Number of actions and numerical fallback threshold.  Sensor columns
        must use the order ``[left, down, right, up]``.
    jit:
        Compile the policy transformation with ``jax.jit``.  Keep this enabled
        for repeated policy calls during a JAX training loop.
    """

    def __init__(self, config: FrozenLakeShieldConfig = FrozenLakeShieldConfig(), *, jit: bool = True):
        jax, _ = _require_jax()
        self.config = config
        self._apply = jax.jit(self._apply_impl) if jit else self._apply_impl

    def _apply_impl(self, base_probs, sensor_values):
        return shield_policy(base_probs, sensor_values, eps=self.config.eps)

    def action_safeties(self, sensor_values):
        """Return per-action safety probabilities in ``[left, down, right, up]`` order."""
        _, jnp = _require_jax()
        values = jnp.asarray(sensor_values)
        if values.shape[-1] != self.config.num_actions:
            raise ValueError(
                f"Expected {self.config.num_actions} sensor/action columns, "
                f"got shape {values.shape}"
            )
        return jnp.clip(1.0 - values, 0.0, 1.0)

    def policy_safety(self, base_probs, sensor_values):
        """Return the probability that a policy draw is safe."""
        _, jnp = _require_jax()
        base = jnp.asarray(base_probs)
        return jnp.sum(base * self.action_safeties(sensor_values), axis=-1)

    def get_shielded_policy(self, base_probs, sensor_values):
        """Mask unsafe actions and renormalize, with a uniform fallback."""
        return self._apply(base_probs, sensor_values)

