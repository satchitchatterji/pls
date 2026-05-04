from __future__ import annotations

import json
import os
from functools import partial

from pls.algorithms.learn import main as learn
from pls.algorithms.pretrain import main as pretrain
from pls.algorithms.evaluate import main as evaluate_policy
from pls.algorithms.ppo_shielded import PPO_shielded
from pls.algorithms.a2c_shielded import A2C_shielded

try:  # optional dependency in full repo
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _noop_sensor(input):
    return input


def _resolve_env_runtime(env_name, algorithm):
    algorithm = algorithm.lower()
    model_map = {
        "ppo": PPO_shielded,
        "a2c": A2C_shielded,
    }
    if algorithm not in model_map:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if env_name == "LineWorldSafety-v0":
        return {
            "model_cls": model_map[algorithm],
            "get_sensor_value_ground_truth": _noop_sensor,
            "custom_callback_class": None,
            "monitor_cls": None,
            "features_extractor_cls": None,
            "observation_net_cls": None,
        }

    # Legacy environment support in full repo, imported lazily.
    if env_name == "Pacman-v0":
        from env_specific_classes.pacman.env_classes import (
            Pacman_Callback,
            Pacman_FeaturesExtractor,
            Pacman_Monitor,
            Pacman_Observation_Net,
        )
        from env_specific_classes.pacman.util import get_ground_wall

        WALL_COLOR = 0.25
        GHOST_COLOR = 0.5
        PACMAN_COLOR = 0.75
        ghost_distance = 1
        return {
            "model_cls": model_map[algorithm],
            "get_sensor_value_ground_truth": partial(
                get_ground_wall, ghost_distance, PACMAN_COLOR, GHOST_COLOR
            ),
            "custom_callback_class": Pacman_Callback,
            "monitor_cls": Pacman_Monitor,
            "features_extractor_cls": Pacman_FeaturesExtractor,
            "observation_net_cls": Pacman_Observation_Net,
        }

    if env_name == "CarRacingPLS-v1":
        from env_specific_classes.carracing.env_classes import (
            Carracing_Callback,
            Carracing_FeaturesExtractor,
            Carracing_Monitor,
            Carracing_Observation_Net,
        )
        from env_specific_classes.carracing.util import get_ground_truth_of_grass

        return {
            "model_cls": model_map[algorithm],
            "get_sensor_value_ground_truth": get_ground_truth_of_grass,
            "custom_callback_class": Carracing_Callback,
            "monitor_cls": Carracing_Monitor,
            "features_extractor_cls": Carracing_FeaturesExtractor,
            "observation_net_cls": Carracing_Observation_Net,
        }

    raise ValueError(f"Unknown env: {env_name}")


def evaluate(config_file, model_at_step, n_test_episodes):
    config_folder = os.path.dirname(config_file)
    with open(config_file) as f:
        config = json.load(f)

    algorithm = config.get("algorithm", config.get("base_policy", "ppo"))
    runtime = _resolve_env_runtime(config["env"], algorithm)

    return evaluate_policy(
        config_folder,
        config,
        model_at_step,
        n_test_episodes,
        runtime["monitor_cls"],
        runtime["model_cls"],
    )


def test(config_file):
    config_folder = os.path.dirname(config_file)
    with open(config_file) as f:
        config = json.load(f)

    config["policy_params"]["total_timesteps"] = 1
    algorithm = config.get("algorithm", config.get("base_policy", "ppo"))
    runtime = _resolve_env_runtime(config["env"], algorithm)

    learn(
        config_folder,
        config,
        runtime["model_cls"],
        runtime["get_sensor_value_ground_truth"],
        runtime["custom_callback_class"],
        runtime["monitor_cls"],
        runtime["features_extractor_cls"],
        runtime["observation_net_cls"],
    )


def train(config_file):
    config_folder = os.path.dirname(config_file)
    with open(config_file) as f:
        config = json.load(f)

    algorithm = config.get("algorithm", config.get("base_policy", "ppo"))
    runtime = _resolve_env_runtime(config["env"], algorithm)

    if config.get("monitor_wandb") and wandb is not None:
        wandb.init(
            project=f"{config['env']}_trial_wandb_integration",
            config=config,
            sync_tensorboard=True,
        )

    learn(
        config_folder,
        config,
        runtime["model_cls"],
        runtime["get_sensor_value_ground_truth"],
        runtime["custom_callback_class"],
        runtime["monitor_cls"],
        runtime["features_extractor_cls"],
        runtime["observation_net_cls"],
    )


def pretrain_observation(
    csv_file,
    img_folder,
    observation_net_folder,
    image_dim,
    downsampling_size,
    net_class,
    num_training_examples,
    epochs,
    labels,
    pretrain_w_extra_labels,
    num_test_examples=200,
):
    net_input_size = image_dim if downsampling_size is None else (image_dim // downsampling_size) ** 2
    net_output_size = len(labels) if not pretrain_w_extra_labels else len(labels) - 2

    pretrain(
        csv_file=csv_file,
        image_folder=img_folder,
        model_folder=observation_net_folder,
        num_training_examples=num_training_examples,
        net_class=net_class,
        net_input_size=net_input_size,
        net_output_size=net_output_size,
        image_dim=image_dim,
        downsampling_size=downsampling_size,
        epochs=epochs,
        keys=labels,
        pretrain_w_extra_labels=pretrain_w_extra_labels,
        num_test_examples=num_test_examples,
    )
