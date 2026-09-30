"""Pretrain the FrozenLake safety sensor with the optional JAX backend.

This is the JAX counterpart to ``frozenlake_pretrain.py``.  It keeps the same
dataset, architecture, output ordering, and BCE-with-logits objective, but
uses JAX transformations and the selected JAX device.  The resulting archive
is a JAX/NumPy checkpoint and is intentionally separate from the Torch ``.pt``
checkpoint because the parameter formats are different.

Run from repository root:
    python examples/frozenlake/frozenlake_jax.py

For Apple Metal, install the optional backend first:
    python -m pip install -e '.[jax-macos]'
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp

from pls.jax.frozenlake import (
    FrozenLakeSensorConfig,
    pretrain_sensor,
    predict_sensor,
    save_sensor,
)
from pls.jax.shields import FrozenLakeJaxShield


# =========================
# Global Experiment Settings
# =========================
SEED = 0
EPOCHS = 120
LEARNING_RATE = 1e-2
CONFIG = FrozenLakeSensorConfig(grid_size=4, hidden_sizes=(16, 16), out_dim=4)
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "images"
CHECKPOINT_PATH = ROOT / "frozenlake_sensor_mlp_jax.npz"
PLOT_PATH = OUTPUT_DIR / "frozenlake_jax_pretrain_loss.png"


def main():
    print("[1/3] Starting JAX FrozenLake sensor pretraining...")
    print(f"      Architecture: 16 -> 16 -> 16 -> 4, epochs={EPOCHS}")
    params, losses = pretrain_sensor(
        seed=SEED,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        config=CONFIG,
    )
    print(f"      Final BCE loss: {losses[-1]:.6f}")

    print("[2/3] Saving JAX checkpoint...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_sensor(CHECKPOINT_PATH, params, CONFIG)
    print(f"      Saved: {CHECKPOINT_PATH}")

    print("[3/3] Saving loss curve...")
    history = pd.DataFrame({"epoch": range(1, len(losses) + 1), "bce_loss": losses})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["bce_loss"], lw=2)
    ax.set_title("FrozenLake JAX Sensor Pretraining")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"      Saved: {PLOT_PATH}")

    print("[4/4] Running a JAX-native shield check...")
    shield = FrozenLakeJaxShield()
    # State 5 is a hole on the default map; the learned sensor predicts which
    # neighboring actions lead into one of the map's holes.
    sensor_values = jnp.asarray(predict_sensor(params, [5], CONFIG))
    base_policy = jnp.full((1, CONFIG.out_dim), 1.0 / CONFIG.out_dim)
    shielded_policy = shield.get_shielded_policy(base_policy, sensor_values)
    print(f"      Sensor device: {sensor_values.device}")
    print(f"      Shield device: {shielded_policy.device}")
    print(f"      Shielded policy: {shielded_policy}")


if __name__ == "__main__":
    main()
