from __future__ import annotations

import numpy as np
import torch as th
from torch.nn import functional as F

from pls.algorithms.dqn_shielded import DQN_shielded


class DoubleDQN_shielded(DQN_shielded):
    """Double-DQN variant of :class:`DQN_shielded` with PLTD safety option.

    Uses online network for action selection and target network for evaluation.
    """

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
                online_next_q = self.q_net(replay_data.next_observations)
                next_actions = online_next_q.argmax(dim=1, keepdim=True)
                target_next_q = self.q_net_target(replay_data.next_observations)
                next_q_values = th.gather(target_next_q, dim=1, index=next_actions.long())
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            all_current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                all_current_q_values, dim=1, index=replay_data.actions.long()
            )

            td_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            safety_loss = self._compute_safety_loss(
                replay_data.observations, all_current_q_values
            )
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
