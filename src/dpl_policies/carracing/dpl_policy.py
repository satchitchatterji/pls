import gym
import torch as th
from torch import nn
import numpy as np
from typing import Union, Tuple
from torch.distributions import Categorical
import time
from stable_baselines3.common.callbacks import ConvertCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

from stable_baselines3.common.type_aliases import (
    GymObs,
    GymStepReturn,
    Schedule,
)
from stable_baselines3.common.preprocessing import is_image_space

from deepproblog.light import DeepProbLogLayer, DeepProbLogLayer_Approx
from dpl_policies.carracing.util import get_ground_truth_of_grass, is_all_grass, get_ground_truth_of_grass2
from os import path
import pickle
from random import random


#
# WALL_COLOR = th.tensor([0] * 3, dtype=th.float32)
# FLOOR_COLOR = th.tensor([1 / 6] * 3, dtype=th.float32)
# BOX_TARGET_COLOR = th.tensor([2 / 6] * 3, dtype=th.float32)
# BOX_ON_TARGET_COLOR = th.tensor([3 / 6] * 3, dtype=th.float32)
# BOX_COLOR = th.tensor([4 / 6] * 3, dtype=th.float32)
#
# PLAYER_COLOR = th.tensor([5 / 6] * 3, dtype=th.float32)
# PLAYER_ON_TARGET_COLOR = th.tensor([1] * 3, dtype=th.float32)
#
# PLAYER_COLORS = th.tensor(([5 / 6] * 3, [1] * 3))
# BOX_COLORS = th.tensor(([3 / 6] * 3, [4 / 6] * 3))
# OBSTABLE_COLORS = th.tensor(([0] * 3, [3 / 6] * 3, [4 / 6] * 3))
#
#
# NEIGHBORS_RELATIVE_LOCS_BOX = [
#     (-1, 0),
#     (1, 0),
#     (0, -1),
#     (0, 1),
#
# ]  # DO NOT CHANGE THE ORDER: up, down, left, right,
# NEIGHBORS_RELATIVE_LOCS_CORNER = [
#     (-2, 0),
#     (2, 0),
#     (0, -2),
#     (0, 2),
# ]  # DO NOT CHANGE THE ORDER

class Carracing_Encoder(nn.Module):
    def __init__(self, input_size, n_actions, shielding_settings, program_path, debug_program_path, folder):
        super(Carracing_Encoder, self).__init__()
        self.input_size = input_size
        self.n_grass_locs = shielding_settings["n_grass_locs"]
        self.n_actions = n_actions
        self.program_path = program_path
        self.debug_program_path = debug_program_path
        self.folder = folder
        self.sensor_noise = shielding_settings["sensor_noise"]
        self.max_num_rejected_samples = shielding_settings["max_num_rejected_samples"]

    def forward(self, x):
        return x


class Carracing_Callback(ConvertCallback):
    def __init__(self, callback):
        super(Carracing_Callback, self).__init__(callback)
        self.is_rollout = False
        self.is_training = False

    def _on_rollout_start(self):
        self.is_rollout = True

    def _on_rollout_end(self):
        self.is_rollout = False

    def _on_training_start(self):
        self.is_training = True

    def _on_training_end(self):
        self.is_training = False

    def _on_step(self):
        return True


class Carracing_Monitor(Monitor):
    def __init__(self, *args, **kwargs):
        super(Carracing_Monitor, self).__init__(*args, **kwargs)

    def reset(self, **kwargs) -> GymObs:
        return super(Carracing_Monitor, self).reset(**kwargs)


    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        if reward > 0:
            ran = random()
            # print(reward)
            if ran > 0.95:
                self.rewards.append(reward)
            else:
                self.rewards.append(-0.1)
        else:
            self.rewards.append(reward)
        # symbolic_state = get_ground_truth_of_grass(th.from_numpy(observation.copy()).unsqueeze(0))
        # violate_constraint = th.all(symbolic_state)
        # TODO: No green panalty
        # all_green = is_all_grass(observation)
        # if all_green: # green penalty
        #     reward -= 0.05
        # self.rewards.append(reward)

        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)

            symbolic_state = get_ground_truth_of_grass2(th.from_numpy(observation.copy()).unsqueeze(0))
            violate_constraint = th.all(symbolic_state)

            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
                "last_r": reward,
                "violate_constraint": violate_constraint,
                "is_success": info["is_success"]
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info


class CustomCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    A CNN based architecture the for PPO features extractor.
    The architecture is a standard multi layer CNN with ReLU activations.

    :param observation_space: Metadata about the observation space to operate over. Assumes shape represents HWC.
    :param features_dim: The number of features to extract from the observations.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        self._features_dim = features_dim
        n_stacked_images, _, _ = observation_space.shape
        self.extractor_network = nn.Sequential(  # Input shape (3, 96, 96)
            nn.Conv2d(n_stacked_images, 8, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(128, features_dim, kernel_size=3, stride=1),  # (128, 3, 3) -> (256, 1, 1)
            nn.ReLU()
        )

    """
    Forward pass through the model.

    :param observations: BCHW tensor representing the states to extract features from.

    Returns:
        Tensor of shape (B,features_dim) representing a compressed view of the input image.
        Intended to be used for policy and 
    """

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.extractor_network(observations).view(-1, self._features_dim)


class Carracing_DPLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            image_encoder: Carracing_Encoder = None,
            alpha=0.5,
            differentiable_shield=True,
            **kwargs
    ):
        super(Carracing_DPLActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule,
                                                             features_extractor_class=CustomCNNFeaturesExtractor,
                                                             **kwargs)
        ###############################

        self.image_encoder = image_encoder
        self.input_size = self.image_encoder.input_size

        self.n_grass_locs = self.image_encoder.n_grass_locs

        self.n_actions = self.image_encoder.n_actions
        self.program_path = self.image_encoder.program_path
        self.debug_program_path = self.image_encoder.debug_program_path
        self.folder = self.image_encoder.folder
        self.sensor_noise = self.image_encoder.sensor_noise
        self.alpha = alpha
        self.differentiable_shield = differentiable_shield
        self.max_num_rejected_samples = self.image_encoder.max_num_rejected_samples

        with open(self.program_path) as f:
            self.program = f.read()
        with open(self.debug_program_path) as f:
            self.debug_program = f.read()

        self.evidences = ["safe_next"]
        # IMPORTANT: THE ORDER OF QUERIES IS THE ORDER OF THE OUTPUT
        self.queries = [
                           "safe_action(do_nothing)",
                           "safe_action(accelerate)",
                           "safe_action(brake)",
                           "safe_action(turn_left)",
                           "safe_action(turn_right)",
                       ][: self.n_actions]

        if self.alpha == 0:
            # NO shielding
            pass
        else:
            # HARD shielding and SOFT shielding
            input_struct = {
                "grass": [i for i in range(self.n_grass_locs)],
                "action": [i for i in range(self.n_grass_locs,
                                            self.n_grass_locs + self.n_actions)],
            }
            action_lst = ["do_nothing", "accelerate", "brake", "turn_left", "turn_right"]

            query_struct = {"safe_action": dict(zip(action_lst[:self.n_actions], range(self.n_actions)))}

            cache_path = path.join(self.folder, "../../../data", "dpl_layer.p")
            self.dpl_layer = self.get_layer(
                cache_path,
                program=self.program, queries=self.queries, evidences=["safe_next"],
                input_struct=input_struct, query_struct=query_struct
            )

        if self.alpha == "learned":
            self.alpha_net = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        # For all settings, calculate "safe_next"
        debug_queries = ["safe_next"]
        debug_query_struct = {"safe_next": 0}
        debug_input_struct = {
            "grass": [i for i in range(self.n_grass_locs)],
            "action": [i for i in range(self.n_grass_locs,
                                        self.n_grass_locs + self.n_actions)]
        }
        cache_path = path.join(self.folder, "../../../data", "query_safety_layer.p")
        self.query_safety_layer = self.get_layer(
            cache_path,
            program=self.program,
            queries=debug_queries,
            evidences=[],
            input_struct=debug_input_struct,
            query_struct=debug_query_struct
        )

        self._build(lr_schedule)

    def get_layer(self, cache_path, program, queries, evidences, input_struct, query_struct):
        if path.exists(cache_path):
            return pickle.load(open(cache_path, "rb"))

        layer = DeepProbLogLayer_Approx(
            program=program, queries=queries, evidences=evidences,
            input_struct=input_struct, query_struct=query_struct
        )
        pickle.dump(layer, open(cache_path, "wb"))
        return layer

    def logging_per_episode(self, mass, object_detect_probs, base_policy, action_lookup):
        abs_safe_next_shielded = self.get_step_safety(
            mass.probs,
            object_detect_probs["ground_truth_grass"]
        )
        abs_safe_next_base = self.get_step_safety(
            base_policy,
            object_detect_probs["ground_truth_grass"]
        )
        return abs_safe_next_shielded, abs_safe_next_base

    def logging_per_step(self, mass, object_detect_probs, base_policy, action_lookup, logger):
        for act in range(self.action_space.n):
            logger.record(
                f"policy/shielded {action_lookup[act]}",
                float(mass.probs[0][act]),
            )
        if object_detect_probs.get("alpha") is not None:
            logger.record(
                f"safety/alpha",
                float(object_detect_probs.get("alpha")),
            )

    def get_step_safety(self, policy_distribution, grass_probs):
        with th.no_grad():
            abs_safe_next = self.query_safety_layer(
                x={
                    "grass": grass_probs,
                    "action": policy_distribution,
                }
            )
            return abs_safe_next["safe_next"]

    def forward(self, x, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        obs = self.image_encoder(x)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        base_actions = distribution.distribution.probs

        with th.no_grad():
            ground_truth_grass = get_ground_truth_of_grass(
                input=x,
            )

            grasses = ground_truth_grass + (self.sensor_noise) * th.randn(ground_truth_grass.shape)
            grasses = th.clamp(grasses, min=0, max=1)

            object_detect_probs = {
                "ground_truth_grass": ground_truth_grass
            }

        if self.alpha == 0:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            object_detect_probs["alpha"] = 0
            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions])

        if not self.differentiable_shield and self.alpha == 1:
            num_rejected_samples = 0
            while True:
                actions = distribution.get_actions(deterministic=deterministic)
                with th.no_grad():
                    # Using problog to model check
                    results = self.query_safety_layer(
                        x={
                            "grass": grasses,
                            "action": th.eye(self.n_actions)[actions],
                        }
                    )
                safe_next = results["safe_next"]
                if not th.any(safe_next.isclose(
                        th.zeros(actions.shape))) or num_rejected_samples > self.max_num_rejected_samples:
                    break
                else:
                    num_rejected_samples += 1
            log_prob = distribution.log_prob(actions)
            object_detect_probs["num_rejected_samples"] = num_rejected_samples
            object_detect_probs["alpha"] = 1
            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions])

        if self.differentiable_shield:
            results = self.dpl_layer(
                x={
                    "grass": grasses,
                    "action": base_actions,
                }
            )

            if self.alpha == "learned":
                alpha = self.alpha_net(obs)
                object_detect_probs["alpha"] = alpha
            else:
                alpha = self.alpha
        else:
            with th.no_grad():
                results = self.dpl_layer(
                    x={
                        "grass": grasses,
                        "action": base_actions,
                    }
                )

            if self.alpha == "learned":
                raise NotImplemented
            else:
                alpha = self.alpha
        object_detect_probs["alpha"] = alpha
        safeast_actions = results["safe_action"]
        actions = alpha * safeast_actions + (1 - alpha) * base_actions

        mass = Categorical(probs=actions)
        if not deterministic:
            actions = mass.sample()
        else:
            actions = th.argmax(mass.probs, dim=1)
        log_prob = mass.log_prob(actions)

        return (actions, values, log_prob, mass, [object_detect_probs, base_actions])

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

        if not self.differentiable_shield and self.alpha == 1:
            obs = self.image_encoder(obs)
            features = self.extract_features(obs)
            latent_pi, latent_vf = self.mlp_extractor(features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)

            return values, log_prob, distribution.entropy()

        _, values, _, mass, _ = self.forward(obs)
        log_prob = mass.log_prob(actions)
        return values, log_prob, mass.entropy()

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        with th.no_grad():
            _actions, values, log_prob, mass, _ = self.forward(observation, deterministic)
            return _actions