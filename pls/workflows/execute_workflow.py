from __future__ import annotations

import json
import os

from pls.algorithms.learn import main as learn
from pls.algorithms.pretrain import main as pretrain
from pls.algorithms.evaluate import main as evaluate_policy
from pls.workflows.registry import resolve_runtime_bundle

try:  # optional dependency in full repo
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _get_runtime_from_config(config: dict):
    algorithm = config.get("algorithm", config.get("base_policy", "ppo"))
    return resolve_runtime_bundle(config["env"], algorithm)


def evaluate(config_file, model_at_step, n_test_episodes):
    config_folder = os.path.dirname(config_file)
    with open(config_file) as f:
        config = json.load(f)

    runtime = _get_runtime_from_config(config)

    return evaluate_policy(
        config_folder,
        config,
        model_at_step,
        n_test_episodes,
        runtime.monitor_cls,
        runtime.model_cls,
    )


def test(config_file):
    config_folder = os.path.dirname(config_file)
    with open(config_file) as f:
        config = json.load(f)

    config["policy_params"]["total_timesteps"] = 1
    runtime = _get_runtime_from_config(config)

    learn(
        config_folder,
        config,
        runtime.model_cls,
        runtime.get_sensor_value_ground_truth,
        runtime.custom_callback_class,
        runtime.monitor_cls,
        runtime.features_extractor_cls,
        runtime.observation_net_cls,
    )


def train(config_file):
    config_folder = os.path.dirname(config_file)
    with open(config_file) as f:
        config = json.load(f)

    runtime = _get_runtime_from_config(config)

    if config.get("monitor_wandb") and wandb is not None:
        wandb.init(
            project=f"{config['env']}_trial_wandb_integration",
            config=config,
            sync_tensorboard=True,
        )

    learn(
        config_folder,
        config,
        runtime.model_cls,
        runtime.get_sensor_value_ground_truth,
        runtime.custom_callback_class,
        runtime.monitor_cls,
        runtime.features_extractor_cls,
        runtime.observation_net_cls,
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
