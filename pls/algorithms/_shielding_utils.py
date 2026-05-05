from __future__ import annotations

from typing import Any, Dict, Optional
import warnings

import numpy as np
import torch as th

from pls.shields.shields import Shield


def build_shield(
    shield_params: Optional[Dict[str, Any]],
    config_folder: Optional[str] = None,
    get_sensor_value_ground_truth=None,
):
    if not shield_params:
        return None
    params = dict(shield_params)
    if config_folder is not None:
        params.setdefault("config_folder", config_folder)
    if get_sensor_value_ground_truth is not None:
        params.setdefault("get_sensor_value_ground_truth", get_sensor_value_ground_truth)
    return Shield(**params)


def shielded_discrete_predict(
    *,
    model,
    observation: np.ndarray | dict[str, np.ndarray],
    state: tuple[np.ndarray, ...] | None,
    episode_start: np.ndarray | None,
    deterministic: bool,
    shield: Shield,
):
    obs_tensor, vectorized_env = model.policy.obs_to_tensor(observation)
    if not isinstance(obs_tensor, th.Tensor):
        obs_tensor = th.as_tensor(obs_tensor, device=model.device)

    with th.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        base_probs = dist.distribution.probs

    sensor_values = shield.get_sensor_values(obs_tensor)
    shielded_probs = shield.get_shielded_policy(base_probs, sensor_values)

    if deterministic:
        actions = th.argmax(shielded_probs, dim=1)
    else:
        actions = th.distributions.Categorical(probs=shielded_probs).sample()

    actions_np = actions.detach().cpu().numpy()
    if not vectorized_env:
        actions_np = actions_np.squeeze()
    return actions_np, state


def action_logits_to_probs(action_tensor: th.Tensor, num_actions: int, temperature: float = 1.0) -> Optional[th.Tensor]:
    """Map a continuous action vector to a differentiable categorical proxy.

    This is a softmax relaxation used for actor-critic methods with Box actions.
    It requires action dimension to match `num_actions`.
    """
    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)
    if action_tensor.ndim != 2:
        return None
    if int(action_tensor.shape[1]) != int(num_actions):
        return None
    return th.softmax(action_tensor / max(float(temperature), 1e-6), dim=1)


def distribution_to_action_probs(distribution, num_actions: int, temperature: float = 1.0) -> Optional[th.Tensor]:
    """Extract action probabilities from an SB3 distribution when possible.

    Priority:
    1) Native categorical probabilities (`distribution.distribution.probs`).
    2) Softmax relaxation of continuous distribution mean.
    """
    if hasattr(distribution, "distribution") and hasattr(distribution.distribution, "probs"):
        return distribution.distribution.probs
    mean_actions = getattr(getattr(distribution, "distribution", None), "mean", None)
    if isinstance(mean_actions, th.Tensor):
        return action_logits_to_probs(mean_actions, num_actions=num_actions, temperature=temperature)
    return None


def compute_policy_safety_loss(
    *,
    shield: Optional[Shield],
    observations: th.Tensor,
    base_actions: th.Tensor,
    eps: float = 1e-8,
) -> th.Tensor:
    """Compute ``-log(P_safe)`` mean from shield outputs."""
    if shield is None:
        return th.tensor(0.0, device=base_actions.device)
    sensor_values = shield.get_sensor_values(observations)
    policy_safeties = shield.get_policy_safety(sensor_values, base_actions).flatten()
    return -th.log(policy_safeties + eps).mean()


def maybe_warn_missing_relaxation(
    *,
    class_name: str,
    shield: Optional[Shield],
    action_dim: int,
    num_actions: Optional[int],
) -> None:
    if shield is None or num_actions is None:
        return
    if int(action_dim) != int(num_actions):
        warnings.warn(
            f"{class_name}: action_dim ({action_dim}) != shield num_actions ({num_actions}). "
            "Safety loss/continuous shielding relaxation will be disabled.",
            stacklevel=2,
        )
