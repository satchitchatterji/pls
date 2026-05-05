"""Minimal end-to-end SPPO example on FrozenLake using a pretrained MLP sensor.

This script is intentionally self-contained:
- no registry setup,
- no external config files,
- shield program defined as a local ProbLog string.

Pedagogical workflow:
1. Create a FrozenLake environment and wrapper for episode metrics.
2. Generate supervised sensor labels directly from map geometry.
3. Pretrain an MLP sensor model on those labels.
4. Wrap the pretrained model as a sensor interface (`predict(obs)`).
5. Train `PPO_shielded` (SPPO) with that sensor.
6. Plot MLP pretraining loss and RL reward/safety curves.
"""

from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from pls.algorithms.ppo_shielded import PPO_shielded

# =========================
# Global Experiment Settings
# =========================
SEED = 0
MAP_NAME = "4x4"
IS_SLIPPERY = True
GRID_SIZE = 4
TOTAL_TIMESTEPS = 20_000
SHIELD_ALPHA = 0.1
MA_WINDOW = 20

# MLP pretraining settings
MLP_HIDDEN_SIZES = (16, 16)
MLP_EPOCHS = 120
MLP_LR = 1e-2
MLP_TEMPERATURE = 1.0
OUTPUT_DIR = Path(__file__).resolve().parent / "images" / "frozenlake"

# FrozenLake 4x4 hole states in default map:
# S F F F
# F H F H
# F F F H
# H F F G
HOLE_STATES = {5, 7, 11, 12}

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
    """Attach episode-level reward and failure proxy to terminal `info`."""

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
    """Collect per-episode reward/failure rows during RL training."""

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
    """Show a tqdm progress bar over RL timesteps."""

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


class FrozenLakeSensorMLP(nn.Module):
    """Small MLP mapping one-hot state features -> 4 sensor probabilities."""

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
    """Sensor wrapper exposing `predict(obs, info=None) -> [batch, 4]` tensor."""

    def __init__(self, model: FrozenLakeSensorMLP):
        self.model = model.eval()

    def predict(self, obs, info=None):
        del info
        arr = np.asarray(obs)
        if arr.ndim == 0:
            arr = np.array([arr])
        arr = arr.astype(np.int64).reshape(-1)
        feats = np.eye(GRID_SIZE * GRID_SIZE, dtype=np.float32)[arr]
        t = th.as_tensor(feats, dtype=th.float32)
        with th.no_grad():
            probs = th.sigmoid(self.model(t)).clamp(0, 1)
        return probs


def next_state(state: int, action: int, grid_size: int = GRID_SIZE) -> int:
    """Deterministic intended next state with boundary clamping."""
    row, col = divmod(int(state), grid_size)
    if action == 0:  # left
        col2, row2 = max(0, col - 1), row
    elif action == 1:  # down
        col2, row2 = col, min(grid_size - 1, row + 1)
    elif action == 2:  # right
        col2, row2 = min(grid_size - 1, col + 1), row
    elif action == 3:  # up
        col2, row2 = col, max(0, row - 1)
    else:
        raise ValueError(action)
    return row2 * grid_size + col2


def build_supervised_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate one-hot state inputs and action-to-hole labels.

    Input features:
    - one-hot state vectors of shape [16, 16].

    Labels (per state):
    - [left_to_hole, down_to_hole, right_to_hole, up_to_hole] in {0,1}^4.
    """
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


def pretrain_mlp_sensor(seed: int = SEED):
    """Pretrain MLP sensor and return sensor wrapper + loss history DataFrame."""
    print("[3/7] Generating supervised sensor dataset...")
    x, y = build_supervised_dataset()
    print(f"      Dataset ready: X shape={x.shape}, y shape={y.shape}")

    print("[4/7] Pretraining MLP sensor...")
    th.manual_seed(seed)
    np.random.seed(seed)

    model = FrozenLakeSensorMLP(
        in_dim=GRID_SIZE * GRID_SIZE,
        hidden_sizes=MLP_HIDDEN_SIZES,
        out_dim=4,
    )
    opt = th.optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = th.nn.BCEWithLogitsLoss()

    xb = th.tensor(x, dtype=th.float32)
    yb = th.tensor(y, dtype=th.float32)

    losses = []
    pbar = tqdm(range(MLP_EPOCHS), desc="Pretraining MLP", unit="epoch")
    for _ in pbar:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    hist = pd.DataFrame(
        {
            "epoch": np.arange(1, MLP_EPOCHS + 1),
            "bce_loss": np.asarray(losses, dtype=np.float32),
        }
    )
    print("      MLP pretraining complete.")
    return FrozenLakePretrainedMLPSensor(model), hist


def moving_mean(x: np.ndarray, window: int = MA_WINDOW) -> np.ndarray:
    """Return moving average with `min_periods=1` for stable early points."""
    if len(x) == 0:
        return x
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()


def main():
    """Run pretrain -> SPPO train -> plot pipeline."""
    print("[1/7] Creating FrozenLake environment...")
    env = gym.make("FrozenLake-v1", map_name=MAP_NAME, is_slippery=IS_SLIPPERY)
    env = EpisodeInfoWrapper(env)
    env.reset(seed=SEED)
    print(f"      Environment ready: {MAP_NAME}, is_slippery={IS_SLIPPERY}")

    print("[2/7] Building in-file shield program...")
    print("      Shield program ready (action-to-hole safety rules).")

    sensor, mlp_hist = pretrain_mlp_sensor(seed=SEED)

    print("[5/7] Constructing SPPO model with pretrained MLP sensor...")
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

    print("[6/7] Training SPPO...")
    cb = RewardSafetyCallback()
    pbar_cb = ProgressBarCallback(total_timesteps=TOTAL_TIMESTEPS)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[cb, pbar_cb])
    print(f"      Training complete. Episodes logged: {len(cb.rows)}")

    print("[7/7] Plotting MLP and RL curves...")
    df = pd.DataFrame(cb.rows)
    if df.empty:
        print("No episodes finished during RL training; plotting only MLP loss.")
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(mlp_hist["epoch"], mlp_hist["bce_loss"], lw=2)
        ax.set_title("MLP Sensor Pretraining Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "frozenlake_mlp_pretrain_only.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"      Saved figure: {out_path}")
        plt.show()
        return

    reward_ma = moving_mean(df["episode_reward"].to_numpy(), window=MA_WINDOW)
    fail_ma = moving_mean(df["episode_failure"].to_numpy(), window=MA_WINDOW)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(mlp_hist["epoch"], mlp_hist["bce_loss"], lw=2, color="tab:blue")
    axes[0].set_title("MLP Sensor Pretraining Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["timestep"], reward_ma, lw=2)
    axes[1].set_title("SPPO on FrozenLake: Reward")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Episode reward (MA)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["timestep"], fail_ma, lw=2, color="tab:red")
    axes[2].set_title("SPPO on FrozenLake: Failure Rate")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Failure rate (MA)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "frozenlake_mlp_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"      Saved figure: {out_path}")
    plt.show()
    print("      Done.")


if __name__ == "__main__":
    main()
