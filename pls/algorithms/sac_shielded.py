from __future__ import annotations

import warnings

import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from pls.algorithms._shielding_utils import (
    action_logits_to_probs,
    build_shield,
    compute_policy_safety_loss,
    maybe_warn_missing_relaxation,
    shielded_discrete_predict,
)


class SAC_shielded(SAC):
    """SAC wrapper with CleanPLS shield-compatible API.

    For environments/policies exposing discrete action distributions, predict-time
    shielding can be applied. Otherwise, this behaves as base SAC.
    """

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
        if self.exploration_shield is not None and not isinstance(self.action_space, spaces.Discrete):
            warnings.warn(
                "SAC_shielded received shield_params but action space is not Discrete; "
                "falling back to base SAC action selection.",
                stacklevel=2,
            )
        if isinstance(self.action_space, spaces.Box):
            num_actions = None
            if self.policy_safety_calculater is not None:
                num_actions = self.policy_safety_calculater.num_actions
            maybe_warn_missing_relaxation(
                class_name=self.__class__.__name__,
                shield=self.policy_safety_calculater,
                action_dim=int(np.prod(self.action_space.shape)),
                num_actions=num_actions,
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

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, safety_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss_base = (ent_coef * log_prob - min_qf_pi).mean()

            safety_loss = th.tensor(0.0, device=actor_loss_base.device)
            if self.policy_safety_calculater is not None and self.alpha > 0.0:
                base_actions = action_logits_to_probs(
                    actions_pi,
                    num_actions=self.policy_safety_calculater.num_actions,
                )
                if base_actions is not None:
                    safety_loss = compute_policy_safety_loss(
                        shield=self.policy_safety_calculater,
                        observations=replay_data.observations,
                        base_actions=base_actions,
                    )
            actor_loss = actor_loss_base + self.alpha * safety_loss

            actor_losses.append(actor_loss.item())
            safety_losses.append(float(safety_loss.item()))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/safety_loss", np.mean(safety_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
