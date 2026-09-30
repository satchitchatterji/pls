"""A device-resident JAX FrozenLake policy-and-shield training pipeline.

This module is intentionally small and research-facing.  It moves every
operation that can reasonably be expressed as array computation onto the
selected JAX device:

* batched FrozenLake transitions,
* one-hot state features,
* frozen MLP sensor inference,
* policy and value networks,
* JAX-native shield masking,
* action sampling,
* generalized advantage estimates,
* PPO-style policy/value/safety loss and Adam updates.

The remaining host work is limited to initializing the experiment and reading
final scalar metrics.  This is an accelerator benchmark/reference pipeline,
not a replacement for the existing SB3 PPO implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .frozenlake import FrozenLakeSensorConfig, _apply, _require_jax, pretrain_sensor, shield_policy


@dataclass(frozen=True)
class FrozenLakeJaxPipelineConfig:
    """Workload and optimization settings for the JAX pipeline benchmark."""

    grid_size: int = 4
    hidden_size: int = 32
    num_actions: int = 4
    num_envs: int = 256
    rollout_steps: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    safety_coef: float = 0.01
    eps: float = 1e-8


HOLES = (5, 7, 11, 12)
GOAL = 15


def _init_layer(key, in_dim: int, out_dim: int):
    jax, jnp = _require_jax()
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return {
        "w": jax.random.uniform(
            key,
            (in_dim, out_dim),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        ),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def init_policy(seed: int, config: FrozenLakeJaxPipelineConfig, device: Any = None):
    """Initialize an actor-critic policy and place it on ``device``."""
    jax, _ = _require_jax()
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    params = {
        "hidden_1": _init_layer(keys[0], config.grid_size**2, config.hidden_size),
        "hidden_2": _init_layer(keys[1], config.hidden_size, config.hidden_size),
        "actor": _init_layer(keys[2], config.hidden_size, config.num_actions),
        "critic": _init_layer(keys[3], config.hidden_size, 1),
    }
    if device is not None:
        params = jax.tree_util.tree_map(lambda value: jax.device_put(value, device), params)
    return params


def _one_hot(states, config: FrozenLakeJaxPipelineConfig):
    _, jnp = _require_jax()
    return jnp.eye(config.grid_size**2, dtype=jnp.float32)[states]


def _is_hole(states):
    _, jnp = _require_jax()
    return (states == HOLES[0]) | (states == HOLES[1]) | (states == HOLES[2]) | (states == HOLES[3])


def transition(states, actions, grid_size: int = 4):
    """Apply deterministic FrozenLake transitions with boundary clamping."""
    _, jnp = _require_jax()
    rows, cols = states // grid_size, states % grid_size
    down = actions == 1
    right = actions == 2
    up = actions == 3
    left = actions == 0
    next_rows = jnp.where(down, jnp.minimum(rows + 1, grid_size - 1), rows)
    next_rows = jnp.where(up, jnp.maximum(next_rows - 1, 0), next_rows)
    next_cols = jnp.where(right, jnp.minimum(cols + 1, grid_size - 1), cols)
    next_cols = jnp.where(left, jnp.maximum(next_cols - 1, 0), next_cols)
    return next_rows * grid_size + next_cols


def policy_forward(policy_params, sensor_params, states, sensor_config, pipeline_config):
    """Compute base policy, shielded policy, values, and policy safety."""
    jax, jnp = _require_jax()
    features = _one_hot(states, pipeline_config)
    hidden = jax_relu(features @ policy_params["hidden_1"]["w"] + policy_params["hidden_1"]["b"])
    hidden = jax_relu(hidden @ policy_params["hidden_2"]["w"] + policy_params["hidden_2"]["b"])
    actor_logits = hidden @ policy_params["actor"]["w"] + policy_params["actor"]["b"]
    values = (hidden @ policy_params["critic"]["w"] + policy_params["critic"]["b"]).squeeze(-1)

    sensor_values = jax_sigmoid(_apply(sensor_params, features))
    base_probs = jax.nn.softmax(actor_logits, axis=-1)
    safety = jnp.clip(1.0 - sensor_values, 0.0, 1.0)
    policy_safety = jnp.sum(base_probs * safety, axis=-1)
    shielded_probs = shield_policy(base_probs, sensor_values, eps=pipeline_config.eps)
    return base_probs, shielded_probs, values, policy_safety


def rollout(policy_params, sensor_params, key, sensor_config, pipeline_config):
    """Collect a fully device-resident vectorized rollout."""
    jax, jnp = _require_jax()
    batch = pipeline_config.num_envs
    initial_states = jnp.zeros((batch,), dtype=jnp.int32)

    def step(carry, _):
        states, current_key = carry
        current_key, action_key = jax.random.split(current_key)
        _, probs, values, policy_safety = policy_forward(
            policy_params, sensor_params, states, sensor_config, pipeline_config
        )
        actions = jax.random.categorical(action_key, jnp.log(probs + pipeline_config.eps), axis=-1)
        next_states = transition(states, actions, pipeline_config.grid_size)
        holes = _is_hole(next_states)
        goals = next_states == GOAL
        dones = holes | goals
        rewards = jnp.where(goals, 1.0, jnp.where(holes, -1.0, -0.01))
        reset_states = jnp.where(dones, 0, next_states)
        indices = jnp.arange(batch)
        log_probs = jnp.log(probs[indices, actions] + pipeline_config.eps)
        transition_data = (states, actions, log_probs, values, rewards, dones, policy_safety, reset_states)
        return (reset_states, current_key), transition_data

    (_, final_key), data = jax.lax.scan(
        step,
        (initial_states, key),
        xs=None,
        length=pipeline_config.rollout_steps,
    )
    return data, final_key


def _gae(data, policy_params, sensor_params, sensor_config, pipeline_config):
    jax, jnp = _require_jax()
    states, actions, old_log_probs, values, rewards, dones, policy_safeties, next_states = data
    flat_next = next_states.reshape(-1)
    next_values = policy_forward(
        policy_params,
        sensor_params,
        flat_next,
        sensor_config,
        pipeline_config,
    )[2].reshape(next_states.shape)
    deltas = rewards + pipeline_config.gamma * (1.0 - dones) * next_values - values

    def reverse_step(carry, inputs):
        delta, done = inputs
        advantage = delta + pipeline_config.gamma * pipeline_config.gae_lambda * (1.0 - done) * carry
        return advantage, advantage

    _, advantages_reversed = jax.lax.scan(
        reverse_step,
        jnp.zeros((pipeline_config.num_envs,), dtype=jnp.float32),
        (deltas[::-1], dones[::-1]),
    )
    advantages = advantages_reversed[::-1]
    returns = advantages + values
    return states, actions, old_log_probs, advantages, returns, policy_safeties


def _loss(policy_params, sensor_params, batch, sensor_config, pipeline_config):
    _, jnp = _require_jax()
    states, actions, old_log_probs, advantages, returns, old_safeties = batch
    flat_states = states.reshape(-1)
    flat_actions = actions.reshape(-1)
    flat_old_log_probs = old_log_probs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)
    flat_old_safeties = old_safeties.reshape(-1)

    mean_advantage = jnp.mean(flat_advantages)
    std_advantage = jnp.std(flat_advantages) + 1e-8
    normalized_advantages = (flat_advantages - mean_advantage) / std_advantage
    base_probs, probs, values, policy_safety = policy_forward(
        policy_params,
        sensor_params,
        flat_states,
        sensor_config,
        pipeline_config,
    )
    indices = jnp.arange(flat_actions.shape[0])
    log_probs = jnp.log(probs[indices, flat_actions] + pipeline_config.eps)
    ratio = jnp.exp(log_probs - flat_old_log_probs)
    clipped_ratio = jnp.clip(
        ratio,
        1.0 - pipeline_config.clip_range,
        1.0 + pipeline_config.clip_range,
    )
    policy_loss = -jnp.mean(jnp.minimum(ratio * normalized_advantages, clipped_ratio * normalized_advantages))
    value_loss = jnp.mean(jnp.square(flat_returns - values))
    entropy = -jnp.mean(jnp.sum(probs * jnp.log(probs + pipeline_config.eps), axis=-1))
    safety_loss = -jnp.mean(jnp.log(policy_safety + pipeline_config.eps))
    loss = (
        policy_loss
        + pipeline_config.value_coef * value_loss
        - pipeline_config.entropy_coef * entropy
        + pipeline_config.safety_coef * safety_loss
    )
    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "safety_loss": safety_loss,
        "mean_policy_safety": jnp.mean(policy_safety),
        "mean_old_safety": jnp.mean(flat_old_safeties),
        "mean_base_probability": jnp.mean(base_probs),
    }
    return loss, metrics


def jax_relu(value):
    """Import-safe ReLU helper used inside transformed functions."""
    _, jnp = _require_jax()
    return jnp.maximum(value, 0.0)


def jax_sigmoid(value):
    """Import-safe sigmoid helper used for frozen sensor inference."""
    _, jnp = _require_jax()
    return jnp.reciprocal(1.0 + jnp.exp(-value))


def _adam_init(params):
    _, jnp = _require_jax()
    return {"m": jax_tree_zeros(params), "v": jax_tree_zeros(params), "step": jnp.asarray(0, dtype=jnp.int32)}


def jax_tree_zeros(tree):
    jax, _ = _require_jax()
    return jax.tree_util.tree_map(lambda value: jax_zeros_like(value), tree)


def jax_zeros_like(value):
    _, jnp = _require_jax()
    return jnp.zeros_like(value)


def _adam_update(params, gradients, state, learning_rate):
    _, jnp = _require_jax()
    step = state["step"] + 1
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    moments = jax_tree_update(state["m"], gradients, beta1)
    velocities = jax_tree_square_update(state["v"], gradients, beta2)
    updates = tree_adam_update(params, moments, velocities, step, learning_rate, beta1, beta2, eps)
    return updates, {"m": moments, "v": velocities, "step": step}


def jax_tree_update(old_tree, gradients, beta):
    jax, _ = _require_jax()
    return jax.tree_util.tree_map(lambda old, grad: beta * old + (1.0 - beta) * grad, old_tree, gradients)


def jax_tree_square_update(old_tree, gradients, beta):
    jax, jnp = _require_jax()
    return jax.tree_util.tree_map(
        lambda old, grad: beta * old + (1.0 - beta) * jnp.square(grad),
        old_tree,
        gradients,
    )


def tree_adam_update(params, moments, velocities, step, learning_rate, beta1, beta2, eps):
    jax, jnp = _require_jax()

    def update(param, moment, velocity):
        moment_hat = moment / (1.0 - beta1**step)
        velocity_hat = velocity / (1.0 - beta2**step)
        return param - learning_rate * moment_hat / (jnp.sqrt(velocity_hat) + eps)

    return jax.tree_util.tree_map(update, params, moments, velocities)


def make_update(sensor_params, sensor_config, pipeline_config):
    """Create a jitted rollout, loss, gradient, and optimizer update."""
    jax, _ = _require_jax()

    def update(policy_params, optimizer_state, key):
        data, next_key = rollout(
            policy_params,
            sensor_params,
            key,
            sensor_config,
            pipeline_config,
        )
        batch = _gae(data, policy_params, sensor_params, sensor_config, pipeline_config)
        loss_and_metrics = lambda current_params: _loss(
            current_params,
            sensor_params,
            batch,
            sensor_config,
            pipeline_config,
        )
        (loss, metrics), gradients = jax.value_and_grad(loss_and_metrics, has_aux=True)(policy_params)
        del loss
        next_params, next_optimizer_state = _adam_update(
            policy_params,
            gradients,
            optimizer_state,
            pipeline_config.learning_rate,
        )
        metrics = dict(metrics)
        metrics["loss"] = metrics["loss"]
        return next_params, next_optimizer_state, next_key, metrics

    return jax.jit(update)


def run_pipeline_benchmark(
    *,
    seed: int = 0,
    updates: int = 100,
    warmup_updates: int = 5,
    sensor_epochs: int = 20,
    sensor_config: FrozenLakeSensorConfig = FrozenLakeSensorConfig(),
    pipeline_config: FrozenLakeJaxPipelineConfig = FrozenLakeJaxPipelineConfig(),
    device: Any = None,
) -> dict[str, Any]:
    """Run the device-resident JAX pipeline and return timing/quality metrics."""
    jax, _ = _require_jax()
    if device is None:
        device = jax.devices()[0]

    sensor_params, sensor_losses = pretrain_sensor(
        seed=seed,
        epochs=sensor_epochs,
        config=sensor_config,
        device=device,
    )
    policy_params = init_policy(seed + 1, pipeline_config, device=device)
    optimizer_state = _adam_init(policy_params)
    optimizer_state = jax.tree_util.tree_map(lambda value: jax.device_put(value, device), optimizer_state)
    key = jax.device_put(jax.random.PRNGKey(seed + 2), device)
    update = make_update(sensor_params, sensor_config, pipeline_config)

    with jax.default_device(device):
        for _ in range(warmup_updates):
            policy_params, optimizer_state, key, metrics = update(policy_params, optimizer_state, key)
        jax.block_until_ready(policy_params)
        start = time.perf_counter()
        for _ in range(updates):
            policy_params, optimizer_state, key, metrics = update(policy_params, optimizer_state, key)
        jax.block_until_ready(policy_params)
        elapsed = time.perf_counter() - start

    return {
        "backend": "jax",
        "device": str(device.platform).lower(),
        "status": "ok",
        "updates": updates,
        "num_envs": pipeline_config.num_envs,
        "rollout_steps": pipeline_config.rollout_steps,
        "transitions": updates * pipeline_config.num_envs * pipeline_config.rollout_steps,
        "seconds": elapsed,
        "update_ms": elapsed / updates * 1000.0,
        "transitions_per_second": updates * pipeline_config.num_envs * pipeline_config.rollout_steps / elapsed,
        "final_loss": float(metrics["loss"]),
        "mean_policy_safety": float(metrics["mean_policy_safety"]),
        "final_sensor_bce": float(sensor_losses[-1]),
        "jax_device": str(device),
        "jax_version": jax.__version__,
    }
