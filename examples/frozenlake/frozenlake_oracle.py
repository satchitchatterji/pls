"""Minimal end-to-end SPPO example on FrozenLake.

This script is intentionally self-contained and does not rely on registry
setup or external config files. It demonstrates the full shielded training
pipeline in five steps:

1. Create a Gymnasium FrozenLake environment.
2. Wrap the env to log per-episode reward and a simple safety proxy.
3. Define a local sensor wrapper that maps state -> safety-relevant signals.
4. Define the shield directly from a ProbLog program string.
5. Train `PPO_shielded` (SPPO), collect metrics, and plot curves.

The safety proxy in plots is:
- `episode_failure = 1` if final episode reward < 1.0 (goal not reached),
- `episode_failure = 0` otherwise.
"""

from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from pls.algorithms.ppo_shielded import PPO_shielded


OUTPUT_DIR = Path(__file__).resolve().parent / "images"


class EpisodeInfoWrapper(gym.Wrapper):
    """Attach episode-level metrics to `info` when an episode ends.

    Added fields at terminal/truncated timesteps:
    - `episode_reward`: cumulative reward in the episode.
    - `episode_failure`: 1.0 if reward < 1.0, else 0.0.
    """

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
    """Collect terminal episode metrics during SB3 training.

    Reads metrics injected by `EpisodeInfoWrapper` and stores per-episode rows
    used later for reward/failure plotting.
    """

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
    """Display a tqdm progress bar over training timesteps."""

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


def moving_mean(x: np.ndarray, window: int = 20) -> np.ndarray:
    """Return a moving average with `min_periods=1` for stable early points."""
    if len(x) == 0:
        return x
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()


class FrozenLakeHoleRiskSensor:
    """Rule-based safety sensor for FrozenLake-4x4.

    Input:
    - scalar or batched discrete state ids.

    Output (`[batch, 4]`):
    - index 0: `left_to_hole`
    - index 1: `down_to_hole`
    - index 2: `right_to_hole`
    - index 3: `up_to_hole`

    Each output is 1.0 if taking the corresponding action from the current
    state leads directly into a hole under deterministic grid transition logic.
    """

    def __init__(self, grid_size: int = 4):
        self.grid_size = int(grid_size)
        self.holes = {5, 7, 11, 12}

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

        def leads_to_hole(action: int) -> th.Tensor:
            ns = self._next_state(s, action)
            return (ns.unsqueeze(1) == hole_ids.unsqueeze(0)).any(dim=1).float()

        return th.stack(
            [
                leads_to_hole(0),  # left
                leads_to_hole(1),  # down
                leads_to_hole(2),  # right
                leads_to_hole(3),  # up
            ],
            dim=1,
        )


def main():
    """Run SPPO training on FrozenLake and plot reward/safety curves."""
    total_timesteps = 20_000
    print("[1/6] Creating FrozenLake environment...")
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env = EpisodeInfoWrapper(env)
    env.reset(seed=0)
    print("      Environment ready: FrozenLake-v1 (4x4, is_slippery=True)")

    print("[2/6] Building shield program (from in-file ProbLog string)...")
    shield_program = """
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
    print("      Shield program loaded with 4 action-risk sensors.")

    print("[3/6] Initializing oracle sensor wrapper...")
    sensor = FrozenLakeHoleRiskSensor(grid_size=4)
    print("      Sensor ready: [left_to_hole, down_to_hole, right_to_hole, up_to_hole]")

    print("[4/6] Constructing SPPO model...")
    shield_params = {
        "num_sensors": 4,
        "num_actions": 4,
        "program_str": shield_program,
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
        "program_str": shield_program,
        "sensor_model": sensor,
        "observation_type": "ground truth",
        "get_sensor_value_ground_truth": lambda x: x,
    }

    model = PPO_shielded(
        env=env,
        seed=0,
        verbose=0,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
        alpha=0.1,
        policy_kwargs=policy_kwargs,
        policy_safety_params=safety_params,
    )
    print("      Model ready: PPO_shielded(alpha=0.1)")

    print("[5/6] Training...")
    cb = RewardSafetyCallback()
    pbar_cb = ProgressBarCallback(total_timesteps=total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=[cb, pbar_cb])
    print(f"      Training complete. Episodes logged: {len(cb.rows)}")

    print("[6/6] Plotting reward and failure curves...")
    df = pd.DataFrame(cb.rows)
    if df.empty:
        print("No episodes finished during training; nothing to plot.")
        return

    reward_ma = moving_mean(df["episode_reward"].to_numpy(), window=20)
    fail_ma = moving_mean(df["episode_failure"].to_numpy(), window=20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["timestep"], reward_ma, lw=2)
    axes[0].set_title("SPPO on FrozenLake: Reward")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Episode reward (MA-20)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["timestep"], fail_ma, lw=2, color="tab:red")
    axes[1].set_title("SPPO on FrozenLake: Failure Rate")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Failure rate (MA-20)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "frozenlake_oracle_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"      Saved figure: {out_path}")
    plt.show()
    print("      Done.")


if __name__ == "__main__":
    main()
