from __future__ import annotations

import os
from pathlib import Path
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


def _resolve_resource_path(config_folder: str, maybe_path: str) -> str:
    p = Path(maybe_path)
    if p.is_absolute():
        return str(p)
    return str((Path(config_folder) / p).resolve())


def _build_sensor_model(config: dict, config_folder: str):
    sensor_cfg = config.get("sensor")
    if not sensor_cfg:
        return None

    if "name" not in sensor_cfg:
        raise ValueError("sensor.name is required when sensor block is provided")

    params = dict(sensor_cfg.get("params", {}))
    params.update(config.get("env_features", {}))

    checkpoint_path = params.get("checkpoint_path")
    if isinstance(checkpoint_path, str):
        params["checkpoint_path"] = _resolve_resource_path(config_folder, checkpoint_path)

    return build_sensor_model(sensor_cfg["name"], **params)


def _prepare_shield_params(raw_params: dict, config_folder: str) -> dict:
    params = dict(raw_params or {})
    if not params:
        return {}

    required = ["num_sensors", "num_actions", "shield_program"]
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"shield params missing required keys: {missing}")

    params["shield_program"] = _resolve_resource_path(config_folder, params["shield_program"])
    if not os.path.exists(params["shield_program"]):
        raise FileNotFoundError(f"shield_program not found: {params['shield_program']}")

    return params


def _validate_preflight(env, sensor_model, shield_params: dict, policy_safety_params: dict):
    # Ensure action size is consistent whenever shielding is configured.
    action_count = getattr(getattr(env, "action_space", None), "n", None)

    for block_name, block in (("shield_params", shield_params), ("policy_safety_params", policy_safety_params)):
        if block:
            if action_count is None:
                raise ValueError(
                    f"{block_name} provided but env action_space is not discrete. "
                    "Shielding currently supports discrete action spaces only."
                )
            expected_actions = int(block["num_actions"])
            if action_count != expected_actions:
                raise ValueError(
                    f"{block_name}.num_actions={expected_actions} does not match env action space size {action_count}."
                )

    if sensor_model is not None and shield_params:
        obs, _ = env.reset()
        preds = sensor_model.predict(obs)
        if preds.ndim != 2:
            raise ValueError(
                f"sensor_model.predict must return rank-2 tensor [batch, num_sensors], got shape {tuple(preds.shape)}"
            )
        out_width = int(preds.shape[1])
        expected = int(shield_params["num_sensors"])
        if out_width != expected:
            raise ValueError(
                f"sensor output width {out_width} does not match shield_params.num_sensors {expected}"
            )


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
        net_arch = dict(
            pi=policy_cfg.get("net_arch_pi", [32, 32]),
            vf=policy_cfg.get("net_arch_vf", [32, 32]),
        )
    else:
        net_arch = policy_cfg["net_arch_shared"] + [
            dict(pi=policy_cfg["net_arch_pi"], vf=policy_cfg["net_arch_vf"])
        ]

    observation_params = dict(config.get("observation_params", {}))
    shield_params = _prepare_shield_params(config.get("shield_params") or {}, config_folder)
    policy_safety_params = _prepare_shield_params(
        config.get("policy_safety_params") or {}, config_folder
    )

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

    _validate_preflight(env, sensor_model, shield_params, policy_safety_params)

    custom_callback = (
        custom_callback_cls(policy_safety_params=policy_safety_params)
        if custom_callback_cls is not None
        else EmptyCallback()
    )
    progress_callback = ProgressBarCallback(policy_cfg["total_timesteps"])

    policy_kwargs = {
        "net_arch": net_arch,
        "activation_fn": nn.ReLU,
        "optimizer_class": th.optim.Adam,
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
    )

    if algorithm in {"ppo"}:
        model_common.update(
            alpha=policy_cfg.get("alpha", 0),
            policy_safety_params=policy_safety_params or None,
        )
        policy_kwargs.update(
            shield_params=shield_params or None,
            config_folder=config_folder,
            get_sensor_value_ground_truth=get_sensor_value_ground_truth,
        )
        model = model_cls(
            n_steps=policy_cfg["n_steps"],
            batch_size=policy_cfg["batch_size"],
            n_epochs=policy_cfg["n_epochs"],
            clip_range=policy_cfg["clip_range"],
            **model_common,
        )
    elif algorithm in {"a2c"}:
        model_common.update(
            alpha=policy_cfg.get("alpha", 0),
            policy_safety_params=policy_safety_params or None,
        )
        policy_kwargs.update(
            shield_params=shield_params or None,
            config_folder=config_folder,
            get_sensor_value_ground_truth=get_sensor_value_ground_truth,
        )
        model = model_cls(
            n_steps=policy_cfg["n_steps"],
            **model_common,
        )
    elif algorithm in {"dqn"}:
        policy_kwargs["net_arch"] = policy_cfg.get("net_arch_pi", [64, 64])
        model_common.update(
            alpha=policy_cfg.get("alpha", 0),
            policy_safety_params=policy_safety_params or None,
            shield_params=shield_params or None,
            differentiable_exploration=policy_cfg.get("differentiable_exploration", False),
            pltd_mode=policy_cfg.get("pltd_mode", "off_policy"),
            config_folder=config_folder,
            get_sensor_value_ground_truth=get_sensor_value_ground_truth,
        )
        model = model_cls(
            buffer_size=policy_cfg.get("buffer_size", int(1e5)),
            learning_starts=policy_cfg.get("learning_starts", 1000),
            batch_size=policy_cfg.get("batch_size", 64),
            train_freq=policy_cfg.get("train_freq", 4),
            gradient_steps=policy_cfg.get("gradient_steps", 1),
            target_update_interval=policy_cfg.get("target_update_interval", 1000),
            **model_common,
        )
    elif algorithm in {"dqn_vanilla"}:
        policy_kwargs["net_arch"] = policy_cfg.get("net_arch_pi", [64, 64])
        model = model_cls(
            buffer_size=policy_cfg.get("buffer_size", int(1e5)),
            learning_starts=policy_cfg.get("learning_starts", 1000),
            batch_size=policy_cfg.get("batch_size", 64),
            train_freq=policy_cfg.get("train_freq", 4),
            gradient_steps=policy_cfg.get("gradient_steps", 1),
            target_update_interval=policy_cfg.get("target_update_interval", 1000),
            **model_common,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.set_random_seed(policy_cfg["seed"])
    model.set_logger(new_logger)

    intermediate_model_path = os.path.join(config_folder, "model_checkpoints")
    checkpoint_callback = CheckpointCallback(
        save_freq=int(5e4), save_path=intermediate_model_path
    )

    wandb_callback = EmptyCallback()
    if config.get("monitor_wandb") and WandbCallback is not None:
        wandb_callback = WandbCallback()

    callbacks = [custom_callback, checkpoint_callback, progress_callback, wandb_callback]
    model.learn(total_timesteps=policy_cfg["total_timesteps"], callback=callbacks)

    model.save(os.path.join(config_folder, "model"))
