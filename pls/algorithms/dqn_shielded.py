from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN

from pls.shields.shields import Shield


class DQN_shielded(DQN):
    """DQN with optional shielding and PLTD safety regularization.

    - If ``differentiable_exploration=False``: applies shielding at action
      selection only (no safety term in TD loss).
    - If ``differentiable_exploration=True`` and ``alpha > 0``: augments TD
      loss with PLTD safety penalty ``-alpha * log P_safe``.
    """

    def __init__(
        self,
        *args,
        shield_params: Optional[Dict[str, Any]] = None,
        alpha: float = 0.0,
        policy_safety_params: Optional[Dict[str, Any]] = None,
        differentiable_exploration: bool = False,
        pltd_mode: str = "off_policy",
        exploration_policy: str = "epsilon_greedy",
        softmax_temperature: float = 1.0,
        config_folder: Optional[str] = None,
        get_sensor_value_ground_truth=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.shield_params = shield_params or {}
        self.policy_safety_params = policy_safety_params or {}
        self.differentiable_exploration = bool(differentiable_exploration)
        self.pltd_mode = pltd_mode
        self.exploration_policy = exploration_policy
        self.softmax_temperature = float(max(softmax_temperature, 1e-6))

        self._use_safety_loss = self.differentiable_exploration and self.alpha > 0.0

        self.exploration_shield = None
        if self.shield_params:
            sp = dict(self.shield_params)
            if config_folder is not None:
                sp.setdefault("config_folder", config_folder)
            if get_sensor_value_ground_truth is not None:
                sp.setdefault("get_sensor_value_ground_truth", get_sensor_value_ground_truth)
            self.exploration_shield = Shield(**sp)

        self.policy_safety_calculater = None
        if self.policy_safety_params:
            ps = dict(self.policy_safety_params)
            if config_folder is not None:
                ps.setdefault("config_folder", config_folder)
            if get_sensor_value_ground_truth is not None:
                ps.setdefault("get_sensor_value_ground_truth", get_sensor_value_ground_truth)
            self.policy_safety_calculater = Shield(**ps)

    def _policy_probs_from_q(
        self,
        q_values: th.Tensor,
        deterministic: bool,
        differentiable: bool,
        eps: Optional[float] = None,
    ) -> th.Tensor:
        n_actions = q_values.shape[1]
        eps_value = 0.0 if deterministic else (float(self.exploration_rate) if eps is None else float(eps))

        if self.exploration_policy == "softmax":
            return th.softmax(q_values / self.softmax_temperature, dim=1)

        if self.exploration_policy != "epsilon_greedy":
            raise ValueError(
                f"Unsupported exploration_policy='{self.exploration_policy}'. "
                "Use 'epsilon_greedy' or 'softmax'."
            )

        if differentiable:
            greedy_probs = th.softmax(q_values / self.softmax_temperature, dim=1)
        else:
            greedy_actions = q_values.argmax(dim=1, keepdim=True)
            greedy_probs = th.zeros_like(q_values)
            greedy_probs.scatter_(1, greedy_actions, 1.0)

        uniform = th.full_like(q_values, fill_value=1.0 / n_actions)
        return (1.0 - eps_value) * greedy_probs + eps_value * uniform

    def _compute_base_action_probs(self, obs_tensor: th.Tensor, deterministic: bool) -> th.Tensor:
        with th.no_grad():
            q_values = self.q_net(obs_tensor)
        return self._policy_probs_from_q(
            q_values=q_values, deterministic=deterministic, differentiable=False
        )

    def _compute_safety_loss(self, observations: th.Tensor, q_values: th.Tensor) -> th.Tensor:
        if not self._use_safety_loss or self.policy_safety_calculater is None:
            return th.tensor(0.0, device=q_values.device)

        base_policy = self._policy_probs_from_q(
            q_values=q_values, deterministic=False, differentiable=True
        )
        sensor_values = self.policy_safety_calculater.get_sensor_values(observations)
        policy_safeties = self.policy_safety_calculater.get_policy_safety(
            sensor_values, base_policy
        ).flatten()
        safety_loss = -th.log(policy_safeties + 1e-8).mean()
        return safety_loss

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        if self.exploration_shield is None:
            return super().predict(observation, state, episode_start, deterministic)

        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)
        if not isinstance(obs_tensor, th.Tensor):
            obs_tensor = th.as_tensor(obs_tensor, device=self.device)

        base_probs = self._compute_base_action_probs(obs_tensor, deterministic=deterministic)
        sensor_values = self.exploration_shield.get_sensor_values(obs_tensor)
        shielded_probs = self.exploration_shield.get_shielded_policy(base_probs, sensor_values)

        if deterministic:
            actions = th.argmax(shielded_probs, dim=1)
        else:
            actions = th.distributions.Categorical(probs=shielded_probs).sample()

        actions_np = actions.detach().cpu().numpy()
        if not vectorized_env:
            actions_np = actions_np.squeeze()
        return actions_np, state

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        td_losses = []
        safety_losses = []
        total_losses = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                if self.pltd_mode == "on_policy":
                    # Approximate SARSA-style next action using current exploration policy.
                    current_next_q = self.q_net(replay_data.next_observations)
                    next_action_probs = self._policy_probs_from_q(
                        q_values=current_next_q,
                        deterministic=False,
                        differentiable=False,
                    )
                    next_actions = th.distributions.Categorical(probs=next_action_probs).sample().reshape(-1, 1)
                    next_q_values = th.gather(next_q_values, dim=1, index=next_actions.long())
                else:
                    next_q_values, _ = next_q_values.max(dim=1)
                    next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            all_current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                all_current_q_values, dim=1, index=replay_data.actions.long()
            )

            td_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            safety_loss = self._compute_safety_loss(
                replay_data.observations, all_current_q_values
            )

            # PLTD objective: TD loss - alpha*log(P_safe) == TD loss + alpha*(-log(P_safe))
            loss = td_loss + self.alpha * safety_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            td_losses.append(td_loss.item())
            safety_losses.append(float(safety_loss.item()))
            total_losses.append(loss.item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/td_loss", float(np.mean(td_losses)))
        self.logger.record("train/safety_loss", float(np.mean(safety_losses)))
        self.logger.record("train/loss", float(np.mean(total_losses)))
