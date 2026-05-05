from __future__ import annotations

import copy
from functools import partial

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from pls.algorithms._shielding_utils import (
    build_shield,
    compute_policy_safety_loss,
    distribution_to_action_probs,
    shielded_discrete_predict,
)

try:  # pragma: no cover - optional dependency
    from sb3_contrib import TRPO
    from sb3_contrib.common.utils import conjugate_gradient_solver
except Exception:  # pragma: no cover
    TRPO = None


if TRPO is None:

    class TRPO_shielded:  # pragma: no cover
        """TRPO shim that explains the missing optional dependency."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TRPO_shielded requires sb3-contrib. Install with `pip install sb3-contrib`."
            )

else:

    class TRPO_shielded(TRPO):
        """TRPO wrapper with CleanPLS shield-compatible API."""

        def __init__(
            self,
            *args,
            alpha: float = 0.0,
            shield_params=None,
            policy_safety_params=None,
            config_folder=None,
            get_sensor_value_ground_truth=None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.alpha = float(alpha)
            self.shield_params = shield_params or {}
            self.policy_safety_params = policy_safety_params or {}
            self.exploration_shield = build_shield(
                self.shield_params,
                config_folder=config_folder,
                get_sensor_value_ground_truth=get_sensor_value_ground_truth,
            )
            self.policy_safety_calculater = build_shield(
                self.policy_safety_params,
                config_folder=config_folder,
                get_sensor_value_ground_truth=get_sensor_value_ground_truth,
            )

        def predict(self, observation, state=None, episode_start=None, deterministic=False):
            if self.exploration_shield is None or not isinstance(self.action_space, spaces.Discrete):
                return super().predict(observation, state, episode_start, deterministic)
            return shielded_discrete_predict(
                model=self,
                observation=observation,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
                shield=self.exploration_shield,
            )

        def _distribution_safety_loss(self, observations: th.Tensor, distribution) -> th.Tensor:
            if self.policy_safety_calculater is None or self.alpha <= 0.0:
                return th.tensor(0.0, device=observations.device)
            base_actions = distribution_to_action_probs(
                distribution,
                num_actions=self.policy_safety_calculater.num_actions,
            )
            if base_actions is None:
                return th.tensor(0.0, device=observations.device)
            return compute_policy_safety_loss(
                shield=self.policy_safety_calculater,
                observations=observations,
                base_actions=base_actions,
            )

        def train(self) -> None:
            """TRPO training with safety-augmented actor objective."""
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)

            policy_objective_values = []
            kl_divergences = []
            line_search_results = []
            value_losses = []
            safety_losses = []

            for rollout_data in self.rollout_buffer.get(batch_size=None):
                if self.sub_sampling_factor > 1:
                    rollout_data = RolloutBufferSamples(
                        rollout_data.observations[:: self.sub_sampling_factor],
                        rollout_data.actions[:: self.sub_sampling_factor],
                        None,  # type: ignore[arg-type]
                        rollout_data.old_log_prob[:: self.sub_sampling_factor],
                        rollout_data.advantages[:: self.sub_sampling_factor],
                        None,  # type: ignore[arg-type]
                    )

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                with th.no_grad():
                    old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

                distribution = self.policy.get_distribution(rollout_data.observations)
                log_prob = distribution.log_prob(actions)

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_objective = (advantages * ratio).mean()
                safety_loss = self._distribution_safety_loss(rollout_data.observations, distribution)
                augmented_policy_objective = policy_objective - self.alpha * safety_loss
                safety_losses.append(float(safety_loss.item()))

                kl_div = kl_divergence(distribution, old_distribution).mean()
                self.policy.optimizer.zero_grad()

                actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(
                    kl_div, augmented_policy_objective
                )

                hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)
                search_direction = conjugate_gradient_solver(
                    hessian_vector_product_fn,
                    policy_objective_gradients,
                    max_iter=self.cg_max_steps,
                )

                line_search_max_step_size = 2 * self.target_kl
                line_search_max_step_size /= float(
                    th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
                )
                line_search_max_step_size = np.sqrt(line_search_max_step_size)

                line_search_backtrack_coeff = 1.0
                original_actor_params = [param.detach().clone() for param in actor_params]

                is_line_search_success = False
                with th.no_grad():
                    for _ in range(self.line_search_max_iter):
                        start_idx = 0
                        for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape, strict=True):
                            n_params = param.numel()
                            param.data = (
                                original_param.data
                                + line_search_backtrack_coeff
                                * line_search_max_step_size
                                * search_direction[start_idx : (start_idx + n_params)].view(shape)
                            )
                            start_idx += n_params

                        distribution = self.policy.get_distribution(rollout_data.observations)
                        log_prob = distribution.log_prob(actions)
                        ratio = th.exp(log_prob - rollout_data.old_log_prob)
                        new_policy_objective = (advantages * ratio).mean()
                        new_safety_loss = self._distribution_safety_loss(rollout_data.observations, distribution)
                        new_augmented_policy_objective = new_policy_objective - self.alpha * new_safety_loss
                        kl_div = kl_divergence(distribution, old_distribution).mean()

                        if (kl_div < self.target_kl) and (new_augmented_policy_objective > augmented_policy_objective):
                            is_line_search_success = True
                            safety_losses.append(float(new_safety_loss.item()))
                            break

                        line_search_backtrack_coeff *= self.line_search_shrinking_factor

                    line_search_results.append(is_line_search_success)

                    if not is_line_search_success:
                        for param, original_param in zip(actor_params, original_actor_params, strict=True):
                            param.data = original_param.data.clone()

                        policy_objective_values.append(augmented_policy_objective.item())
                        kl_divergences.append(0.0)
                    else:
                        policy_objective_values.append(new_augmented_policy_objective.item())
                        kl_divergences.append(kl_div.item())

            for _ in range(self.n_critic_updates):
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    values_pred = self.policy.predict_values(rollout_data.observations)
                    value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                    value_losses.append(value_loss.item())

                    self.policy.optimizer.zero_grad()
                    value_loss.backward()
                    for param in actor_params:
                        param.grad = None
                    self.policy.optimizer.step()

            self._n_updates += 1
            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            self.logger.record("train/policy_objective", np.mean(policy_objective_values))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
            self.logger.record("train/safety_loss", np.mean(safety_losses) if len(safety_losses) > 0 else 0.0)
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/is_line_search_success", np.mean(line_search_results))
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
