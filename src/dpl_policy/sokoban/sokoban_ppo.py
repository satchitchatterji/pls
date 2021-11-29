from torch import nn
from deepproblog.light import DeepProbLogLayer
import torch as th
from dpl_policy.sokoban.util import get_ground_relatives
from typing import Any, Dict, Optional, Type, Union, List, Tuple, Generator, NamedTuple
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
import gym
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    FlattenExtractor,
)
from stable_baselines3.common.distributions import make_proba_distribution
from torch.distributions import Categorical
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import (
    GymEnv,
    Schedule,
    MaybeCallback,
)
from stable_baselines3 import PPO
import time
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance
from gym import spaces
from torch.nn import functional as F
from stable_baselines3.common.vec_env import VecNormalize

# WALL_COLOR = 0.25
# GHOST_COLOR = 0.5
# PACMAN_COLOR = 0.75
# FOOD_COLOR = 1

WALL_COLOR = 0
FLOOR_COLOR = 1/6
BOX_TARGET_COLOR = 2/6
BOX_ON_TARGET_COLOR = 3/6
BOX_COLOR = 4/6
PLAYER_COLOR = 5/6
PLAYER_ON_TARGET_COLOR = 1



NEIGHBORS_RELATIVE_LOCS_BOX = [(-1, 0), (0, -1), (0, 1), (1, 0)]

NEIGHBORS_RELATIVE_LOCS_WALL = \
    [
        (-3, 0), (-2, -1), (-2, 0), (-2, 1), (-1, -2),
        (-1,-1), (-1,0), (-1,1), (-1,2), (0,-3),
        (0,-2), (0,-1), (0,1), (0,2), (0,3),
        (1,-2), (1,-1), (1,0), (1,1), (1,2),
        (2,-1), (2,0), (2,1), (3,0)
    ]

#     [
#     ( 0,  3), (-1, 2), (0, 2), (1, 2), (-2, 1),
#     (-1,  1), ( 0, 1), (1, 1), (2, 1), (-3, 0),
#     (-2,  0), (-1, 0), (1, 0), (2, 0), (3, 0),
#     (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
#     (-1, -2), (0, -2), (1, -2), (0, -3)
# ]

NEIGHBORS_RELATIVE_LOCS_TARGET = [( -2,0), (0,-2), (0, 2), (2, 0)]

class Sokoban_Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        n_actions,
        shielding_settings,
        program_path
    ):
        super(Sokoban_Encoder, self).__init__()
        self.input_size = input_size
        self.shield = shielding_settings["shield"]
        self.detect_boxes = shielding_settings["detect_boxes"]
        self.detect_walls = shielding_settings["detect_walls"]
        self.detect_targets = shielding_settings["detect_targets"]
        self.n_actions = n_actions
        self.program_path = program_path

    def forward(self, x):
        xx = th.flatten(x, 1)
        return xx


class Sokoban_DPLRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    boxes_error: th.Tensor
    walls_errors: th.Tensor
    targets_error: th.Tensor


class Sokoban_DPLRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        self.boxes_errors = None
        self.walls_errors = None
        self.targets_errors = None
        super(Sokoban_DPLRolloutBuffer, self).__init__(*args, **kwargs)

    def reset(self) -> None:
        self.boxes_errors = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.walls_errors = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.targets_errors = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super(Sokoban_DPLRolloutBuffer, self).reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        errors
    ) -> None:
        boxes_error = errors[0]
        walls_error = errors[1]
        targets_error = errors[2]
        self.boxes_errors[self.pos] = boxes_error.clone().cpu().numpy()
        self.walls_errors[self.pos] = walls_error.clone().cpu().numpy()
        self.targets_errors[self.pos] = targets_error.clone().cpu().numpy()
        super(Sokoban_DPLRolloutBuffer, self).add(
            obs, action, reward, episode_start, value, log_prob
        )

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[Sokoban_DPLRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "boxes_errors",
                "walls_errors",
                "targets_errors",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Sokoban_DPLRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.boxes_errors[batch_inds].flatten(),
            self.walls_errors[batch_inds].flatten(),
            self.targets_errors[batch_inds].flatten(),
        )
        return Sokoban_DPLRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class Sokoban_DPLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        image_encoder: Sokoban_Encoder = None,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(
            action_space, use_sde=use_sde, dist_kwargs=dist_kwargs
        )

        # self._build(lr_schedule)
        ###############################

        self.image_encoder = image_encoder
        self.input_size = self.image_encoder.input_size
        self.shield = self.image_encoder.shield

        self.detect_boxes = self.image_encoder.detect_boxes
        self.detect_walls = self.image_encoder.detect_walls
        self.detect_targets = self.image_encoder.detect_targets

        self.n_actions = self.image_encoder.n_actions
        self.program_path = self.image_encoder.program_path

        if self.detect_boxes:
            self.box_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                # nn.Softmax()
                nn.Sigmoid(),  # TODO : add a flag
            )
        if self.detect_walls:
            self.wall_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 24),
                # nn.Softmax()
                nn.Sigmoid(),
            )
        if self.detect_targets:
            self.target_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                # nn.Softmax()
                nn.Sigmoid(),
            )

        if self.shield:
            with open(self.program_path) as f:
                self.program = f.read()

            self.queries = [
                "safe_action(no_op)",
                "safe_action(push_up)",
                "safe_action(push_down)",
                "safe_action(push_left)",
                "safe_action(push_right)",
                "safe_action(move_up)",
                "safe_action(move_down)",
                "safe_action(move_left)",
                "safe_action(move_right)",
                "safe_next",
            ]

            # self.evidences = [
            #     "safe_next"
            # ]

            self.dpl_layer = DeepProbLogLayer(
                program=self.program, queries=self.queries#, evidences=self.evidences
            )

        self._build(lr_schedule)

    def forward(self, x, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        obs = self.image_encoder(x)

        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # print(obs)
        # print(latent_pi)

        distribution = self._get_action_dist_from_latent(
            latent_pi, latent_sde=latent_sde
        )

        if not self.shield:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            return (
                actions,
                values,
                log_prob,
                distribution,
                [th.zeros((1, 1)), th.zeros((1, 1)), th.zeros((1, 1))]
            )

        box_ground_relative = get_ground_relatives(
            x[0],
            [PLAYER_COLOR, PLAYER_ON_TARGET_COLOR],
            [BOX_ON_TARGET_COLOR, BOX_COLOR],
            NEIGHBORS_RELATIVE_LOCS_BOX,
            out_of_boundary_value=False
        )
        wall_ground_relative = get_ground_relatives(
            x[0],
            [PLAYER_COLOR, PLAYER_ON_TARGET_COLOR],
            [WALL_COLOR],
            NEIGHBORS_RELATIVE_LOCS_WALL,
            out_of_boundary_value=True
        )
        target_ground_relative = get_ground_relatives(
            x[0],
            [PLAYER_COLOR, PLAYER_ON_TARGET_COLOR],
            [BOX_TARGET_COLOR, BOX_ON_TARGET_COLOR, PLAYER_ON_TARGET_COLOR],
            NEIGHBORS_RELATIVE_LOCS_TARGET,
            out_of_boundary_value=False
        )
        boxes = self.box_layer(obs) if self.detect_boxes else box_ground_relative
        walls = self.wall_layer(obs) if self.detect_walls else wall_ground_relative
        targets = self.target_layer(obs) if self.detect_targets else target_ground_relative

        base_actions = distribution.distribution.probs
        results = self.dpl_layer(
            x={"box": boxes, "wall": walls, "target": targets, "action": base_actions}
        )

        # It's possible there is no safe actions, in that case safe_next is zero
        safe_next = results["safe_next"]
        safe_actions = results["safe_action"] / safe_next
        # When safe_next is zero, we need to use base_actions
        actions = th.where(abs(safe_next)<1e-6, base_actions, safe_actions)

        # print(safe_actions, results["safe_action"], base_actions, safe_next)

        mass = Categorical(probs=actions)
        actions = mass.sample()
        log_prob = mass.log_prob(actions)

        with th.no_grad():
            boxes_error = (
                (box_ground_relative - boxes).abs().sum(dim=1).reshape((-1, 1))
            )
            walls_error = (
                (wall_ground_relative - walls).abs().sum(dim=1).reshape((-1, 1))
            )
            targets_error = (
                (target_ground_relative - targets).abs().sum(dim=1).reshape((-1, 1))
            )

        return actions, values, log_prob, mass, [boxes_error, walls_error, targets_error]

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        _actions, values, log_prob, mass, errors = self.forward(obs)

        log_prob = mass.log_prob(actions)
        return values, log_prob, mass.entropy(), errors


class Sokoban_DPLPPO(PPO):
    def __init__(self, *args, **kwargs):

        super(Sokoban_DPLPPO, self).__init__(*args, **kwargs)
        buffer_cls = Sokoban_DPLRolloutBuffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def _setup_model(self) -> None:
        super(Sokoban_DPLPPO, self)._setup_model()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/ep_rew_mean",
                        safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    )
                    self.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    )
                    self.logger.record(
                        "rollout/ep_last_r_mean",
                        safe_mean(
                            [ep_info["last_r"] for ep_info in self.ep_info_buffer]
                        ),
                    )
                self.logger.record("time/fps", fps)
                self.logger.record(
                    "time/time_elapsed",
                    int(time.time() - self.start_time),
                    exclude="tensorboard",
                )
                self.logger.record(
                    "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                )
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                (
                    actions,
                    values,
                    log_probs,
                    mass,
                    errors
                ) = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            (
                new_obs,
                rewards,
                dones,
                infos
             ) = env.step(clipped_actions)
            for e in env.envs:
                if e.env.render_or_not:
                    e.env.render()

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                errors
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _, _, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, _ = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record(
            "train/boxes_errors", np.mean(self.rollout_buffer.boxes_errors.flatten())
        )
        self.logger.record(
            "train/walls_errors", np.mean(self.rollout_buffer.walls_errors.flatten())
        )
        self.logger.record(
            "train/targets_errors", np.mean(self.rollout_buffer.targets_errors.flatten())
        )


# class DPLPolicyGradientPolicy(OnPolicyAlgorithm):
#     """
#     TODO: An attempt to run policy gradient in stable-baselines
#     """
#     def __init__(
#         self,
#         policy: Union[str, Type[ActorCriticPolicy]],
#         env: Union[GymEnv, str],
#         learning_rate: Union[float, Schedule],
#         n_steps: int,
#         gamma: float,
#         gae_lambda: float,
#         ent_coef: float,
#         vf_coef: float,
#         max_grad_norm: float,
#         use_sde: bool,
#         sde_sample_freq: int,
#         policy_base: Type[BasePolicy] = ActorCriticPolicy,
#         tensorboard_log: Optional[str] = None,
#         create_eval_env: bool = False,
#         monitor_wrapper: bool = True,
#         policy_kwargs: Optional[Dict[str, Any]] = None,
#         verbose: int = 0,
#         seed: Optional[int] = None,
#         device: Union[th.device, str] = "auto",
#         _init_setup_model: bool = True,
#         supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
#         image_encoder: Sokoban_Encoder = None,
#     ):
#         super(DPLPolicyGradientPolicy, self).__init__(
#             policy=policy,
#             env=env,
#             learning_rate=learning_rate,
#             n_steps=n_steps,
#             gamma=gamma,
#             gae_lambda=gae_lambda,
#             ent_coef=ent_coef,
#             vf_coef=vf_coef,
#             max_grad_norm=max_grad_norm,
#             use_sde=use_sde,
#             sde_sample_freq=sde_sample_freq,
#             policy_base=policy_base,
#             tensorboard_log=tensorboard_log,
#             create_eval_env=create_eval_env,
#             monitor_wrapper=monitor_wrapper,
#             policy_kwargs=policy_kwargs,
#             verbose=verbose,
#             seed=seed,
#             device=device,
#             _init_setup_model=_init_setup_model,
#             supported_action_spaces=supported_action_spaces,
#         )
#
#         self.image_encoder = image_encoder
#         # self.logger = self.image_encoder.logger
#         self.input_size = self.image_encoder.input_size
#         self.shield = self.image_encoder.shield
#         self.detect_boxes = self.image_encoder.detect_boxes
#         self.detect_walls = self.image_encoder.detect_walls
#         self.detect_targets = self.image_encoder.detect_targets
#         self.n_actions = self.image_encoder.n_actions
#         self.program_path = self.image_encoder.program_path
#
#         # self.base_policy_layer = self.policy
#
#         if self.detect_boxes:
#             self.ghost_layer = nn.Sequential(
#                 nn.Linear(self.input_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 4),
#                 nn.Sigmoid(),  # TODO : add a flag
#             )
#         if self.detect_walls:
#             self.wall_layer = nn.Sequential(
#                 nn.Linear(self.input_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 24),
#                 nn.Sigmoid(),
#             )
#         if self.detect_targets:
#             self.target_layer = nn.Sequential(
#                 nn.Linear(self.input_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 4),
#                 nn.Sigmoid(),
#             )
#
#         if self.shield:
#             with open(self.program_path) as f:
#                 self.program = f.read()
#
#             self.queries = [
#                 "safe_action(no_op)",
#                 "safe_action(push_up)",
#                 "safe_action(push_down)",
#                 "safe_action(push_left)",
#                 "safe_action(push_right)",
#                 "safe_action(move_up)",
#                 "safe_action(move_down)",
#                 "safe_action(move_left)",
#                 "safe_action(move_right)",
#
#                 "safe_next",
#             ]
#
#
#
#             self.dpl_layer = DeepProbLogLayer(
#                 program=self.program, queries=self.queries
#             )
#
#     def forward(self, x, deterministic: bool = False):
#         """
#         Forward pass in all the networks (actor and critic)
#
#         :param obs: Observation
#         :param deterministic: Whether to sample or use deterministic actions
#         :return: action, value and log probability of the action
#         """
#         obs = self.image_encoder(x)
#
#         latent_pi, latent_vf, latent_sde = self._get_latent(obs)
#         # Evaluate the values for the given observations
#         values = self.value_net(latent_vf)
#         distribution = self._get_action_dist_from_latent(
#             latent_pi, latent_sde=latent_sde
#         )
#
#         if not self.shield:
#             actions = distribution.get_actions(deterministic=deterministic)
#             log_prob = distribution.log_prob(actions)
#             return actions, values, log_prob
#
#         ghosts_ground_relative = get_ground_wall(x[0], PACMAN_COLOR, GHOST_COLOR)
#         wall_ground_relative = get_ground_wall(x[0], PACMAN_COLOR, WALL_COLOR)
#         ghosts = self.ghost_layer(obs) if self.detect_ghosts else ghosts_ground_relative
#         walls = self.wall_layer(obs) if self.detect_walls else wall_ground_relative
#
#         base_actions = distribution.distribution.probs
#         results = self.dpl_layer(
#             x={"ghost": ghosts, "wall": walls, "action": base_actions}
#         )
#
#         actions = results["safe_action"]
#         safe_next = results["safe_next"]
#
#         actions = actions / safe_next
#
#         mass = Categorical(probs=actions)
#         actions = mass.sample()
#         log_prob = mass.log_prob(actions)
#
#         return actions, values, log_prob
