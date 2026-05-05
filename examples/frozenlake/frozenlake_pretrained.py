"""Train SPPO on FrozenLake using a loadable pretrained `.pt` sensor file.

This tutorial-style script is self-contained (no registry/config files):
1. Load a pretrained sensor checkpoint from `frozenlake_pretrain.py`.
2. Wrap it as a `predict(obs, info=None)` sensor interface.
3. Define a shield from an in-file ProbLog string.
4. Train `PPO_shielded` and plot reward/safety curves.
"""

from __future__ import annotations

from pathlib import Path
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from pls.algorithms.ppo_shielded import PPO_shielded

try:
    from frozenlake_pretrain import FrozenLakeSensorMLP  # local tutorial module
except ModuleNotFoundError:
    # Allows execution from repository root via: python examples/frozenlake/frozenlake_pretrained.py
    sys.path.append(str(Path(__file__).resolve().parent))
    from frozenlake_pretrain import FrozenLakeSensorMLP  # type: ignore[no-redef]

# =========================
# Global Experiment Settings
# =========================
SEED = 0
MAP_NAME = "4x4"
IS_SLIPPERY = True
GRID_SIZE = 4
TOTAL_TIMESTEPS = 25_000
SHIELD_ALPHA = 0.1
MA_WINDOW = 20

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "images"
CHECKPOINT_PATH = ROOT / "frozenlake_sensor_mlp.pt"

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


class EpisodeInfoWrapper(gym.Wrapper):
    """Attach episode-level reward/failure proxy to terminal `info`."""

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


class RewardSafetyCallback(BaseCallback):
    """Collect per-episode reward/failure metrics during training."""

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
    """Display a tqdm progress bar over RL timesteps."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self._pbar = None

    def _on_training_start(self):
        self._pbar = tqdm(total=self.total_timesteps, desc="Training SPPO", unit="ts")

    def _on_step(self) -> bool:
        if self._pbar is not None:
            self._pbar.update(1)
        return True

    def _on_training_end(self):
        if self._pbar is not None:
            self._pbar.close()


class FrozenLakeCheckpointSensor:
    """Load a `.pt` checkpoint and expose `predict(obs, info=None)`."""

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


def moving_mean(x: np.ndarray, window: int = MA_WINDOW) -> np.ndarray:
    """Return moving average with `min_periods=1`."""
    if len(x) == 0:
        return x
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()


def main():
    print("[1/6] Checking pretrained checkpoint...")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run: python examples/frozenlake/frozenlake_pretrain.py"
        )
    print(f"      Found checkpoint: {CHECKPOINT_PATH}")

    print("[2/6] Creating FrozenLake environment...")
    env = gym.make("FrozenLake-v1", map_name=MAP_NAME, is_slippery=IS_SLIPPERY)
    env = EpisodeInfoWrapper(env)
    env.reset(seed=SEED)
    print(f"      Environment ready: {MAP_NAME}, is_slippery={IS_SLIPPERY}")

    print("[3/6] Loading pretrained sensor model...")
    sensor = FrozenLakeCheckpointSensor(CHECKPOINT_PATH)
    print("      Sensor ready from .pt file.")

    print("[4/6] Constructing SPPO model...")
    shield_params = {
        "num_sensors": 4,
        "num_actions": 4,
        "program_str": SHIELD_PROGRAM,
        "sensor_model": sensor,
        "observation_type": "ground truth",
    }
    policy_kwargs = {
        "shield_params": shield_params,
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "get_sensor_value_ground_truth": lambda x: x,
    }
    safety_params = {
        "num_sensors": 4,
        "num_actions": 4,
        "program_str": SHIELD_PROGRAM,
        "sensor_model": sensor,
        "observation_type": "ground truth",
        "get_sensor_value_ground_truth": lambda x: x,
    }

    model = PPO_shielded(
        env=env,
        seed=SEED,
        verbose=0,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
        alpha=SHIELD_ALPHA,
        policy_kwargs=policy_kwargs,
        policy_safety_params=safety_params,
    )
    print(f"      Model ready: PPO_shielded(alpha={SHIELD_ALPHA})")

    print("[5/6] Training SPPO...")
    cb = RewardSafetyCallback()
    pbar_cb = ProgressBarCallback(total_timesteps=TOTAL_TIMESTEPS)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[cb, pbar_cb])
    print(f"      Training complete. Episodes logged: {len(cb.rows)}")

    print("[6/6] Plotting reward and failure curves...")
    df = pd.DataFrame(cb.rows)
    if df.empty:
        print("No episodes finished during training; nothing to plot.")
        return

    reward_ma = moving_mean(df["episode_reward"].to_numpy(), window=MA_WINDOW)
    fail_ma = moving_mean(df["episode_failure"].to_numpy(), window=MA_WINDOW)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["timestep"], reward_ma, lw=2)
    axes[0].set_title("SPPO (Pretrained .pt Sensor): Reward")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Episode reward (MA)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["timestep"], fail_ma, lw=2, color="tab:red")
    axes[1].set_title("SPPO (Pretrained .pt Sensor): Failure Rate")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Failure rate (MA)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "frozenlake_pretrained_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"      Saved figure: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
