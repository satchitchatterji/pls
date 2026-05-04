from __future__ import annotations

import os
from typing import Optional, Type

import gymnasium as gym

import torch as th
from torch import nn
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from pls.sensors.registry import build_sensor_model

try:
    from wandb.integration.sb3 import WandbCallback
except ImportError:  # pragma: no cover
    WandbCallback = None

from tqdm import tqdm


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self):
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


class EmptyCallback(BaseCallback):
    def _on_training_start(self):
        pass

    def _on_step(self):
        return True

    def _on_training_end(self):
        pass


class SimpleMonitorWrapper:
    """Compatibility wrapper for old monitor API."""

    def __call__(self, env, allow_early_resets=False, **kwargs):
        del allow_early_resets, kwargs
        return env


def _build_sensor_model(config: dict, config_folder: str):
    sensor_cfg = config.get("sensor")
    if not sensor_cfg:
        return None
    supported_types = {"rule_based", "mlp_pretrained"}
    sensor_type = sensor_cfg.get("type")
    if sensor_type not in supported_types:
        raise ValueError(f"Unsupported sensor.type: {sensor_type}")

    params = dict(sensor_cfg.get("params", {}))
    params.update(config.get("env_features", {}))
    checkpoint_path = params.get("checkpoint_path")
    if isinstance(checkpoint_path, str) and not os.path.isabs(checkpoint_path):
        params["checkpoint_path"] = os.path.normpath(
            os.path.join(config_folder, checkpoint_path)
        )
    return build_sensor_model(sensor_cfg["name"], **params)


def main(
    config_folder,
    config,
    model_cls,
    get_sensor_value_ground_truth=None,
    custom_callback_cls: Optional[Type[BaseCallback]] = None,
    monitor_cls=None,
    features_extractor_cls=None,
    observation_net_cls=None,
):
    del observation_net_cls

    policy_cfg = config["policy_params"]
    algorithm = config.get("algorithm", config.get("base_policy", "ppo")).lower()

    if features_extractor_cls is None:
        net_arch = dict(pi=policy_cfg.get("net_arch_pi", [32, 32]), vf=policy_cfg.get("net_arch_vf", [32, 32]))
    else:
        net_arch = policy_cfg["net_arch_shared"] + [
            dict(pi=policy_cfg["net_arch_pi"], vf=policy_cfg["net_arch_vf"])
        ]

    observation_params = dict(config.get("observation_params", {}))
    shield_params = dict(config.get("shield_params") or {})
    policy_safety_params = dict(config.get("policy_safety_params") or {})

    sensor_model = _build_sensor_model(config, config_folder)

    if shield_params:
        shield_params.update(observation_params)
        if sensor_model is not None:
            shield_params["sensor_model"] = sensor_model

    if policy_safety_params:
        policy_safety_params["config_folder"] = config_folder
        policy_safety_params["get_sensor_value_ground_truth"] = get_sensor_value_ground_truth
        policy_safety_params.update(observation_params)
        if sensor_model is not None:
            policy_safety_params["sensor_model"] = sensor_model

    new_logger = configure(config_folder, ["log", "tensorboard"])

    env = gym.make(config["env"], **config.get("env_features", {}))
    monitor = monitor_cls or SimpleMonitorWrapper()
    env = monitor(env, allow_early_resets=False, **(config.get("monitor_features") or {}))

    custom_callback = (
        custom_callback_cls(policy_safety_params=policy_safety_params)
        if custom_callback_cls is not None
        else EmptyCallback()
    )
    progress_callback = ProgressBarCallback(policy_cfg["total_timesteps"])

    policy_kwargs = {
        "shield_params": shield_params or None,
        "net_arch": net_arch,
        "activation_fn": nn.ReLU,
        "optimizer_class": th.optim.Adam,
        "config_folder": config_folder,
        "get_sensor_value_ground_truth": get_sensor_value_ground_truth,
    }
    if features_extractor_cls is not None:
        policy_kwargs["features_extractor_class"] = features_extractor_cls

    model_common = dict(
        env=env,
        learning_rate=policy_cfg["learning_rate"],
        gamma=policy_cfg["gamma"],
        tensorboard_log=config_folder,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=policy_cfg["seed"],
        _init_setup_model=True,
        alpha=policy_cfg.get("alpha", 0),
        policy_safety_params=policy_safety_params or None,
    )

    if algorithm in {"ppo"}:
        model = model_cls(
            n_steps=policy_cfg["n_steps"],
            batch_size=policy_cfg["batch_size"],
            n_epochs=policy_cfg["n_epochs"],
            clip_range=policy_cfg["clip_range"],
            **model_common,
        )
    elif algorithm in {"a2c"}:
        model = model_cls(
            n_steps=policy_cfg["n_steps"],
            **model_common,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.set_random_seed(policy_cfg["seed"])
    model.set_logger(new_logger)

    intermediate_model_path = os.path.join(config_folder, "model_checkpoints")
    checkpoint_callback = CheckpointCallback(save_freq=int(5e4), save_path=intermediate_model_path)

    wandb_callback = EmptyCallback()
    if config.get("monitor_wandb") and WandbCallback is not None:
        wandb_callback = WandbCallback()

    callbacks = [custom_callback, checkpoint_callback, progress_callback, wandb_callback]
    model.learn(total_timesteps=policy_cfg["total_timesteps"], callback=callbacks)

    model.save(os.path.join(config_folder, "model"))
