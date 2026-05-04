from torch.nn import functional as F
import torch as th
import numpy as np
from gymnasium import spaces

from stable_baselines3 import A2C
from stable_baselines3.common.utils import explained_variance

from pls.algorithms.ppo_shielded import ActorCriticPolicy_shielded
from pls.shields.shields import Shield


class A2C_shielded(A2C):
    """A2C with optional shielded policy safety loss."""

    def __init__(
        self,
        *args,
        policy=ActorCriticPolicy_shielded,
        alpha=0,
        policy_safety_params=None,
        **kwargs,
    ):
        super().__init__(*args, policy=policy, **kwargs)
        self.alpha = alpha
        self.policy_safety_calculater = (
            Shield(**policy_safety_params) if policy_safety_params else None
        )

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions
            )
            values = values.flatten()

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(advantages * log_prob).mean()
            value_loss = F.mse_loss(rollout_data.returns, values)

            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            safety_loss = th.tensor(0.0, device=values.device)
            if self.policy_safety_calculater is not None:
                policy_safeties = self.policy_safety_calculater.get_policy_safety(
                    self.policy.info["sensor_value"],
                    self.policy.info["base_policy"],
                ).flatten()
                safety_loss = -th.log(policy_safeties + 1e-8).mean()

            loss = (
                policy_loss
                + self.ent_coef * entropy_loss
                + self.vf_coef * value_loss
                + self.alpha * safety_loss
            )

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/safety_loss", safety_loss.item())
        self.logger.record("train/loss", loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
