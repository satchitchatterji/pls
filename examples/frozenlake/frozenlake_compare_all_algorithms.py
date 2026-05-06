"""Compare FrozenLake algorithms with standardized shielding (root-runnable).

This script compares base vs shielded variants using two sensor sources:
- base
- sbase(oracle)
- sbase(mlp)

Design goals:
1. One shared ProbLog shield program across shielded variants.
2. One shared shield config contract (`num_sensors`, `num_actions`, alpha).
3. Reproducible runs with explicit seeds and saved CSV metrics.
4. Informative prints + tqdm loading bars during training.

Run from repository root:
    python examples/frozenlake/frozenlake_compare_all_algorithms.py

Optional quick overrides:
    CLEANPLS_SEEDS=0 CLEANPLS_TOTAL_TIMESTEPS=2000 python examples/frozenlake/frozenlake_compare_all_algorithms.py
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from pls.algorithms.a2c_shielded import A2C_shielded
from pls.algorithms.ddpg_shielded import DDPG_shielded
from pls.algorithms.dqn import DQN_standard
from pls.algorithms.dqn_shielded import DQN_shielded
from pls.algorithms.double_dqn_shielded import DoubleDQN_shielded
from pls.algorithms.ppo_shielded import PPO_shielded
from pls.algorithms.rainbow_shielded import Rainbow_shielded
from pls.algorithms.sac_shielded import SAC_shielded
from pls.algorithms.td3_shielded import TD3_shielded
from pls.algorithms.trpo_shielded import TRPO_shielded

try:
    from sb3_contrib import TRPO
except Exception:
    TRPO = None


# =========================
# Global Experiment Settings
# =========================
def _parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


SEEDS = _parse_seeds(os.getenv("CLEANPLS_SEEDS", "0,1"))
TOTAL_TIMESTEPS = int(os.getenv("CLEANPLS_TOTAL_TIMESTEPS", "12000"))
MAP_NAME = "4x4"
IS_SLIPPERY = False
GRID_SIZE = 4
HOLE_STATES = {5, 7, 11, 12}
ROLLING_WINDOW = 20
SHIELD_ALPHA = 0.1

# Shared DQN-style options for consistency
PLTD_MODE = "off_policy"
EXPLORATION_POLICY = "epsilon_greedy"
DIFFERENTIABLE_EXPLORATION = False
SOFTMAX_TEMPERATURE = 1.0

# MLP sensor pretraining
MLP_HIDDEN_SIZES = (16, 16)
MLP_EPOCHS = 120
MLP_LR = 1e-2

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "images"
CHECKPOINT_PATH = ROOT / "frozenlake_sensor_mlp.pt"

RAW_METRICS_CSV = OUTPUT_DIR / "frozenlake_all_algorithms_raw_metrics.csv"
SUMMARY_CSV = OUTPUT_DIR / "frozenlake_all_algorithms_summary.csv"
SKIPPED_CSV = OUTPUT_DIR / "frozenlake_all_algorithms_skipped.csv"
PLOT_PATH = OUTPUT_DIR / "frozenlake_all_algorithms_curves.png"
SUMMARY_PLOT_PATH = OUTPUT_DIR / "frozenlake_all_algorithms_grouped_summary.png"
MLP_PRETRAIN_PLOT = OUTPUT_DIR / "frozenlake_all_algorithms_mlp_pretrain_loss.png"

NOOP_SENSOR_FN = lambda x: x

SHIELD_PROGRAM = """
action(0)::action(left);
action(1)::action(down);
action(2)::action(right);
action(3)::action(up).

sensor_value(0)::left_to_hole.
sensor_value(1)::down_to_hole.
sensor_value(2)::right_to_hole.
sensor_value(3)::up_to_hole.

unsafe_next :- left_to_hole, action(left).
unsafe_next :- down_to_hole, action(down).
unsafe_next :- right_to_hole, action(right).
unsafe_next :- up_to_hole, action(up).

safe_next :- \\+ unsafe_next.
""".strip()

GROUP_BASE = "base"
GROUP_ORACLE = "sbase(oracle)"
GROUP_MLP = "sbase(mlp)"


@dataclass
class Variant:
    family: str
    group: str
    label: str
    make_model: Callable
    compatible_with_frozenlake: bool
    skip_reason: str = ""


class EpisodeInfoWrapper(gym.Wrapper):
    """Attach episode reward/failure to terminal `info` for logging."""

    def __init__(self, env):
        super().__init__(env)
        self._ep_reward = 0.0

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += float(reward)
        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = float(self._ep_reward)
            info["episode_failure"] = float(self._ep_reward < 1.0)
        return obs, reward, terminated, truncated, info


class EpisodeMetricsCallback(BaseCallback):
    """Collect per-episode reward/failure rows from env infos."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if isinstance(info, dict) and "episode_reward" in info:
                self.rows.append(
                    {
                        "timestep": int(self.num_timesteps),
                        "episode_reward": float(info["episode_reward"]),
                        "episode_failure": float(info["episode_failure"]),
                    }
                )
        return True


class ProgressBarCallback(BaseCallback):
    """Per-variant training progress bar."""

    def __init__(self, total_timesteps: int, label: str):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.label = label
        self._pbar = None

    def _on_training_start(self):
        self._pbar = tqdm(total=self.total_timesteps, desc=self.label, unit="ts")

    def _on_step(self) -> bool:
        if self._pbar is not None:
            self._pbar.update(1)
        return True

    def _on_training_end(self):
        if self._pbar is not None:
            self._pbar.close()


class FrozenLakeHoleRiskSensor:
    """Deterministic oracle sensor: state -> [left,down,right,up]-to-hole probs."""

    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = int(grid_size)
        self.holes = set(HOLE_STATES)

    def _next_state(self, s: th.Tensor, action: int) -> th.Tensor:
        row = th.div(s, self.grid_size, rounding_mode="floor")
        col = s % self.grid_size
        if action == 0:  # left
            col2 = th.clamp(col - 1, min=0)
            row2 = row
        elif action == 1:  # down
            row2 = th.clamp(row + 1, max=self.grid_size - 1)
            col2 = col
        elif action == 2:  # right
            col2 = th.clamp(col + 1, max=self.grid_size - 1)
            row2 = row
        elif action == 3:  # up
            row2 = th.clamp(row - 1, min=0)
            col2 = col
        else:
            raise ValueError(action)
        return row2 * self.grid_size + col2

    def predict(self, obs, info=None):
        del info
        if isinstance(obs, th.Tensor):
            s = obs.detach().long().reshape(-1)
        else:
            s = th.as_tensor(np.asarray(obs), dtype=th.long).reshape(-1)

        hole_ids = th.as_tensor(sorted(self.holes), dtype=th.long, device=s.device)

        def to_hole(action: int) -> th.Tensor:
            ns = self._next_state(s, action)
            return (ns.unsqueeze(1) == hole_ids.unsqueeze(0)).any(dim=1).float()

        return th.stack([to_hole(0), to_hole(1), to_hole(2), to_hole(3)], dim=1)


class FrozenLakeSensorMLP(nn.Module):
    """Small MLP for action-to-hole sensor prediction from one-hot state."""

    def __init__(self, in_dim=16, hidden_sizes=(16, 16), out_dim=4):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FrozenLakePretrainedMLPSensor:
    """Load MLP checkpoint and expose `predict(obs, info=None)` sensor API."""

    def __init__(self, checkpoint_path: Path):
        payload = th.load(checkpoint_path, map_location="cpu")
        hidden_sizes = tuple(payload["hidden_sizes"])
        in_dim = int(payload["grid_size"]) * int(payload["grid_size"])
        out_dim = int(payload.get("out_dim", 4))

        model = FrozenLakeSensorMLP(in_dim=in_dim, hidden_sizes=hidden_sizes, out_dim=out_dim)
        model.load_state_dict(payload["state_dict"])
        self.model = model.eval()
        self.grid_size = int(payload["grid_size"])

    def predict(self, obs, info=None):
        del info
        arr = np.asarray(obs)
        if arr.ndim == 0:
            arr = np.array([arr])
        arr = arr.astype(np.int64).reshape(-1)
        feats = np.eye(self.grid_size * self.grid_size, dtype=np.float32)[arr]
        t = th.as_tensor(feats, dtype=th.float32)
        with th.no_grad():
            probs = th.sigmoid(self.model(t)).clamp(0, 1)
        return probs


def next_state(state: int, action: int, grid_size: int = GRID_SIZE) -> int:
    row, col = divmod(int(state), grid_size)
    if action == 0:
        col2, row2 = max(0, col - 1), row
    elif action == 1:
        col2, row2 = col, min(grid_size - 1, row + 1)
    elif action == 2:
        col2, row2 = min(grid_size - 1, col + 1), row
    elif action == 3:
        col2, row2 = col, max(0, row - 1)
    else:
        raise ValueError(action)
    return row2 * grid_size + col2


def build_supervised_dataset() -> tuple[np.ndarray, np.ndarray]:
    n_states = GRID_SIZE * GRID_SIZE
    states = np.arange(n_states, dtype=np.int64)
    x = np.eye(n_states, dtype=np.float32)[states]

    labels = []
    for s in states:
        labels.append(
            [
                float(next_state(s, 0) in HOLE_STATES),
                float(next_state(s, 1) in HOLE_STATES),
                float(next_state(s, 2) in HOLE_STATES),
                float(next_state(s, 3) in HOLE_STATES),
            ]
        )
    y = np.asarray(labels, dtype=np.float32)
    return x, y


def ensure_pretrained_sensor_checkpoint() -> FrozenLakePretrainedMLPSensor:
    """Load `.pt` sensor if present, otherwise pretrain and save it."""
    if CHECKPOINT_PATH.exists():
        print(f"[Sensor] Found pretrained checkpoint: {CHECKPOINT_PATH}")
        return FrozenLakePretrainedMLPSensor(CHECKPOINT_PATH)

    print("[Sensor] No checkpoint found. Pretraining MLP sensor now...")
    th.manual_seed(0)
    np.random.seed(0)

    x, y = build_supervised_dataset()
    model = FrozenLakeSensorMLP(in_dim=GRID_SIZE * GRID_SIZE, hidden_sizes=MLP_HIDDEN_SIZES, out_dim=4)
    opt = th.optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = th.nn.BCEWithLogitsLoss()

    xb = th.tensor(x, dtype=th.float32)
    yb = th.tensor(y, dtype=th.float32)

    losses = []
    pbar = tqdm(range(MLP_EPOCHS), desc="Pretraining FrozenLake MLP sensor", unit="epoch")
    for _ in pbar:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "grid_size": GRID_SIZE,
        "hidden_sizes": tuple(MLP_HIDDEN_SIZES),
        "out_dim": 4,
    }
    th.save(payload, CHECKPOINT_PATH)
    print(f"[Sensor] Saved pretrained checkpoint: {CHECKPOINT_PATH}")

    hist = pd.DataFrame({"epoch": np.arange(1, MLP_EPOCHS + 1), "bce_loss": losses})
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(hist["epoch"], hist["bce_loss"], lw=2)
    ax.set_title("MLP Sensor Pretraining Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MLP_PRETRAIN_PLOT, dpi=150, bbox_inches="tight")
    print(f"[Sensor] Saved pretraining curve: {MLP_PRETRAIN_PLOT}")

    return FrozenLakePretrainedMLPSensor(CHECKPOINT_PATH)


def make_env(seed: int):
    env = gym.make("FrozenLake-v1", map_name=MAP_NAME, is_slippery=IS_SLIPPERY)
    env = EpisodeInfoWrapper(env)
    env.reset(seed=seed)
    return env


def build_shield_params(sensor_model: object) -> dict:
    return {
        "num_sensors": 4,
        "num_actions": 4,
        "program_str": SHIELD_PROGRAM,
        "sensor_model": sensor_model,
        "observation_type": "ground truth",
    }


def build_policy_kwargs(shield_params: dict) -> dict:
    return {
        "shield_params": dict(shield_params),
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "get_sensor_value_ground_truth": NOOP_SENSOR_FN,
    }


def build_variants(oracle_sensor, mlp_sensor):
    oracle_params = build_shield_params(oracle_sensor)
    mlp_params = build_shield_params(mlp_sensor)
    oracle_policy_kwargs = build_policy_kwargs(oracle_params)
    mlp_policy_kwargs = build_policy_kwargs(mlp_params)

    def ppo_base(env, seed):
        return PPO(
            "MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.99,
            clip_range=0.2,
            policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
        )

    def ppo_shield(env, seed, group: str):
        params = oracle_params if group == GROUP_ORACLE else mlp_params
        pkwargs = oracle_policy_kwargs if group == GROUP_ORACLE else mlp_policy_kwargs
        return PPO_shielded(
            env=env,
            seed=seed,
            verbose=0,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.99,
            clip_range=0.2,
            alpha=SHIELD_ALPHA,
            policy_kwargs=pkwargs,
            policy_safety_params=dict(params),
        )

    def a2c_base(env, seed):
        return A2C(
            "MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            n_steps=64,
            learning_rate=7e-4,
            gamma=0.99,
            policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
        )

    def a2c_shield(env, seed, group: str):
        params = oracle_params if group == GROUP_ORACLE else mlp_params
        pkwargs = oracle_policy_kwargs if group == GROUP_ORACLE else mlp_policy_kwargs
        return A2C_shielded(
            env=env,
            seed=seed,
            verbose=0,
            n_steps=64,
            learning_rate=7e-4,
            gamma=0.99,
            alpha=SHIELD_ALPHA,
            policy_kwargs=pkwargs,
            policy_safety_params=dict(params),
        )

    def trpo_base(env, seed):
        if TRPO is None:
            raise RuntimeError("sb3_contrib.TRPO is unavailable; install sb3-contrib.")
        return TRPO(
            "MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            n_steps=128,
            batch_size=64,
            learning_rate=1e-3,
            gamma=0.99,
            policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
        )

    def trpo_shield(env, seed, group: str):
        if TRPO is None:
            raise RuntimeError("sb3_contrib.TRPO is unavailable; install sb3-contrib.")
        params = oracle_params if group == GROUP_ORACLE else mlp_params
        return TRPO_shielded(
            "MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            n_steps=128,
            batch_size=64,
            learning_rate=1e-3,
            gamma=0.99,
            alpha=SHIELD_ALPHA,
            policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
            policy_safety_params=dict(params),
        )

    def dqn_base_cls(cls, env, seed):
        return cls(
            "MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            learning_rate=1e-3,
            gamma=0.99,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=64,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            policy_kwargs={"net_arch": [64, 64]},
        )

    def dqn_shield_cls(cls, env, seed, group: str):
        params = oracle_params if group == GROUP_ORACLE else mlp_params
        return cls(
            "MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            learning_rate=1e-3,
            gamma=0.99,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=64,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            policy_kwargs={"net_arch": [64, 64]},
            alpha=SHIELD_ALPHA,
            shield_params=dict(params),
            policy_safety_params=dict(params),
            pltd_mode=PLTD_MODE,
            exploration_policy=EXPLORATION_POLICY,
            differentiable_exploration=DIFFERENTIABLE_EXPLORATION,
            softmax_temperature=SOFTMAX_TEMPERATURE,
        )

    variants: list[Variant] = []

    def add_family(family: str, base_fn: Callable, shield_fn: Callable, compatible: bool = True, skip_reason: str = ""):
        variants.append(Variant(family, GROUP_BASE, f"{family}/{GROUP_BASE}", base_fn, compatible, skip_reason))
        variants.append(
            Variant(
                family,
                GROUP_ORACLE,
                f"{family}/{GROUP_ORACLE}",
                lambda env, seed, _shield_fn=shield_fn: _shield_fn(env, seed, GROUP_ORACLE),
                compatible,
                skip_reason,
            )
        )
        variants.append(
            Variant(
                family,
                GROUP_MLP,
                f"{family}/{GROUP_MLP}",
                lambda env, seed, _shield_fn=shield_fn: _shield_fn(env, seed, GROUP_MLP),
                compatible,
                skip_reason,
            )
        )

    add_family("A2C", a2c_base, a2c_shield, compatible=True)
    add_family("PPO", ppo_base, ppo_shield, compatible=True)
    add_family("TRPO", trpo_base, trpo_shield, compatible=(TRPO is not None), skip_reason="sb3-contrib missing")
    add_family("DQN", lambda env, seed: dqn_base_cls(DQN_standard, env, seed), lambda env, seed, group: dqn_shield_cls(DQN_shielded, env, seed, group), compatible=True)
    add_family("DoubleDQN", lambda env, seed: dqn_base_cls(DoubleDQN_shielded, env, seed), lambda env, seed, group: dqn_shield_cls(DoubleDQN_shielded, env, seed, group), compatible=True)
    add_family("Rainbow", lambda env, seed: dqn_base_cls(Rainbow_shielded, env, seed), lambda env, seed, group: dqn_shield_cls(Rainbow_shielded, env, seed, group), compatible=True)

    # Kept for completeness; these are incompatible with FrozenLake's Discrete action space.
    add_family(
        "SAC",
        lambda env, seed: SAC_shielded("MlpPolicy", env=env, seed=seed, verbose=0),
        lambda env, seed, group: SAC_shielded(
            "MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            alpha=SHIELD_ALPHA,
            shield_params=dict(oracle_params if group == GROUP_ORACLE else mlp_params),
            policy_safety_params=dict(oracle_params if group == GROUP_ORACLE else mlp_params),
        ),
        compatible=False,
        skip_reason="SAC in SB3 requires continuous (Box) action spaces.",
    )
    add_family(
        "TD3",
        lambda env, seed: TD3_shielded("MlpPolicy", env=env, seed=seed, verbose=0),
        lambda env, seed, group: TD3_shielded(
            "MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            alpha=SHIELD_ALPHA,
            shield_params=dict(oracle_params if group == GROUP_ORACLE else mlp_params),
            policy_safety_params=dict(oracle_params if group == GROUP_ORACLE else mlp_params),
        ),
        compatible=False,
        skip_reason="TD3 in SB3 requires continuous (Box) action spaces.",
    )
    add_family(
        "DDPG",
        lambda env, seed: DDPG_shielded("MlpPolicy", env=env, seed=seed, verbose=0),
        lambda env, seed, group: DDPG_shielded(
            "MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            alpha=SHIELD_ALPHA,
            shield_params=dict(oracle_params if group == GROUP_ORACLE else mlp_params),
            policy_safety_params=dict(oracle_params if group == GROUP_ORACLE else mlp_params),
        ),
        compatible=False,
        skip_reason="DDPG in SB3 requires continuous (Box) action spaces.",
    )

    return variants


def train_one_variant(variant: Variant, seed: int) -> pd.DataFrame:
    print(f"\n=== Training {variant.label} (seed={seed}) ===")
    env = make_env(seed)
    model = variant.make_model(env, seed)
    metrics_cb = EpisodeMetricsCallback()
    pbar_cb = ProgressBarCallback(total_timesteps=TOTAL_TIMESTEPS, label=f"{variant.label} s={seed}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[metrics_cb, pbar_cb])

    rows = metrics_cb.rows
    env.close()

    if not rows:
        print(f"{variant.label} (seed={seed}): no episode completions logged.")
        return pd.DataFrame(
            columns=[
                "variant",
                "family",
                "group",
                "seed",
                "episode_idx",
                "timestep",
                "episode_reward",
                "episode_failure",
            ]
        )

    df = pd.DataFrame(rows)
    df.insert(0, "episode_idx", np.arange(len(df), dtype=int))
    df.insert(0, "seed", int(seed))
    df.insert(0, "group", variant.group)
    df.insert(0, "family", variant.family)
    df.insert(0, "variant", variant.label)

    print(
        f"{variant.label} (seed={seed}) done: "
        f"episodes={len(df)}, mean_reward={df['episode_reward'].mean():.4f}, "
        f"failure_rate={df['episode_failure'].mean():.4f}"
    )
    return df


def make_plots(all_runs: pd.DataFrame, summary_df: pd.DataFrame):
    print("\n[Plot] Building comparative plots...")

    # Plot 1: all variant training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for variant, part in all_runs.groupby("variant"):
        part = part.sort_values(["seed", "episode_idx"]).copy()
        part["reward_ma"] = part.groupby("seed")["episode_reward"].transform(
            lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        part["failure_ma"] = part.groupby("seed")["episode_failure"].transform(
            lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )

        agg_reward = part.groupby("episode_idx", as_index=False)["reward_ma"].mean()
        agg_fail = part.groupby("episode_idx", as_index=False)["failure_ma"].mean()

        axes[0].plot(agg_reward["episode_idx"], agg_reward["reward_ma"], lw=2, label=variant)
        axes[1].plot(agg_fail["episode_idx"], agg_fail["failure_ma"], lw=2, label=variant)

    axes[0].set_title(f"FrozenLake Reward (MA-{ROLLING_WINDOW})")
    axes[0].set_xlabel("Episode index")
    axes[0].set_ylabel("Episode reward")
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f"FrozenLake Failure Rate (MA-{ROLLING_WINDOW})")
    axes[1].set_xlabel("Episode index")
    axes[1].set_ylabel("Failure rate")
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")

    # Plot 2: grouped bars per family with base/oracle/mlp + std error bars over seeds.
    seed_level = (
        all_runs.groupby(["family", "group", "variant", "seed"], as_index=False)
        .agg(
            mean_episode_reward=("episode_reward", "mean"),
            failure_rate=("episode_failure", "mean"),
        )
    )
    grouped_stats = (
        seed_level.groupby(["family", "group"], as_index=False)
        .agg(
            reward_mean=("mean_episode_reward", "mean"),
            reward_std=("mean_episode_reward", "std"),
            fail_mean=("failure_rate", "mean"),
            fail_std=("failure_rate", "std"),
        )
    )
    grouped = grouped_stats.pivot(index="family", columns="group", values="reward_mean")
    grouped_err = grouped_stats.pivot(index="family", columns="group", values="reward_std")
    grouped_fail = grouped_stats.pivot(index="family", columns="group", values="fail_mean")
    grouped_fail_err = grouped_stats.pivot(index="family", columns="group", values="fail_std")

    family_order = [f for f in ["A2C", "PPO", "TRPO", "DQN", "DoubleDQN", "Rainbow"] if f in grouped.index]
    grouped = grouped.reindex(family_order)
    grouped_fail = grouped_fail.reindex(family_order)

    groups = [GROUP_BASE, GROUP_ORACLE, GROUP_MLP]
    x = np.arange(len(grouped.index))
    width = 0.25

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))

    for i, g in enumerate(groups):
        y_reward = grouped[g] if g in grouped.columns else np.nan
        y_reward_err = grouped_err[g] if g in grouped_err.columns else np.nan
        y_fail = grouped_fail[g] if g in grouped_fail.columns else np.nan
        y_fail_err = grouped_fail_err[g] if g in grouped_fail_err.columns else np.nan
        axes2[0].bar(
            x + (i - 1) * width,
            y_reward,
            width=width,
            yerr=y_reward_err,
            capsize=4,
            label=g,
        )
        axes2[1].bar(
            x + (i - 1) * width,
            y_fail,
            width=width,
            yerr=y_fail_err,
            capsize=4,
            label=g,
        )

    axes2[0].set_title("Mean Episode Reward by Family and Group")
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(grouped.index, rotation=0)
    axes2[0].set_ylabel("Mean episode reward")
    axes2[0].grid(axis="y", alpha=0.3)

    axes2[1].set_title("Failure Rate by Family and Group")
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(grouped_fail.index, rotation=0)
    axes2[1].set_ylabel("Failure rate")
    axes2[1].grid(axis="y", alpha=0.3)

    axes2[0].legend(title="group")
    axes2[1].legend(title="group")
    fig2.tight_layout()
    fig2.savefig(SUMMARY_PLOT_PATH, dpi=150, bbox_inches="tight")

    print(f"Saved plot: {PLOT_PATH}")
    print(f"Saved plot: {SUMMARY_PLOT_PATH}")


def main():
    print("=" * 88)
    print("FrozenLake All-Algorithm Comparison (base / sbase(oracle) / sbase(mlp))")
    print("=" * 88)
    print(f"Map={MAP_NAME}, is_slippery={IS_SLIPPERY}, seeds={SEEDS}, timesteps={TOTAL_TIMESTEPS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Setup] Building standardized sensors and shield resources...")
    oracle_sensor = FrozenLakeHoleRiskSensor(grid_size=GRID_SIZE)
    mlp_sensor = ensure_pretrained_sensor_checkpoint()
    print("[Setup] Shield standardization across all shielded runs:")
    print("- same ProbLog program")
    print("- same num_sensors/num_actions")
    print("- same alpha")
    print("- same interface; only sensor source changes: oracle vs mlp")

    variants = build_variants(oracle_sensor, mlp_sensor)

    skipped_rows = []
    run_frames = []

    for variant in variants:
        if not variant.compatible_with_frozenlake:
            reason = variant.skip_reason or "Not compatible with FrozenLake action space."
            print(f"[Skip] {variant.label}: {reason}")
            skipped_rows.append({"variant": variant.label, "family": variant.family, "group": variant.group, "reason": reason})
            continue

        for seed in SEEDS:
            try:
                run_frames.append(train_one_variant(variant, seed))
            except Exception as exc:
                reason = f"Runtime error: {type(exc).__name__}: {exc}"
                print(f"[Skip] {variant.label} (seed={seed}) due to error: {reason}")
                skipped_rows.append({"variant": variant.label, "family": variant.family, "group": variant.group, "reason": reason})

    if not run_frames:
        print("No successful training runs. Exiting.")
        pd.DataFrame(skipped_rows).to_csv(SKIPPED_CSV, index=False)
        print(f"Saved skip reasons: {SKIPPED_CSV}")
        return

    all_runs = pd.concat(run_frames, ignore_index=True)
    all_runs.to_csv(RAW_METRICS_CSV, index=False)
    print(f"\nSaved raw metrics: {RAW_METRICS_CSV}")

    summary = (
        all_runs.groupby(["family", "group", "variant"], as_index=False)
        .agg(
            mean_episode_reward=("episode_reward", "mean"),
            failure_rate=("episode_failure", "mean"),
            episodes=("episode_idx", "count"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["family", "group"])
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary: {SUMMARY_CSV}")

    if skipped_rows:
        pd.DataFrame(skipped_rows).drop_duplicates().to_csv(SKIPPED_CSV, index=False)
        print(f"Saved skipped variants: {SKIPPED_CSV}")

    print("\nSummary table:")
    print(summary.to_string(index=False))

    make_plots(all_runs, summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
