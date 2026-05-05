"""Pretrain a FrozenLake MLP sensor and save it as a loadable `.pt` file.

This script is self-contained and intentionally simple:
1. Build supervised labels from FrozenLake map geometry.
2. Train an MLP that predicts action-to-hole risk sensors.
3. Save checkpoint + metadata for later SPPO training.
4. Plot and save pretraining loss curve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from tqdm.auto import tqdm

# =========================
# Global Experiment Settings
# =========================
SEED = 0
GRID_SIZE = 4
HOLE_STATES = {5, 7, 11, 12}
MLP_HIDDEN_SIZES = (16, 16)
MLP_EPOCHS = 120
MLP_LR = 1e-2

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "images"
CHECKPOINT_PATH = ROOT / "frozenlake_sensor_mlp.pt"


class FrozenLakeSensorMLP(nn.Module):
    """Small MLP mapping one-hot state features -> 4 sensor logits."""

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
    """Generate one-hot state inputs and action-to-hole labels."""
    n_states = GRID_SIZE * GRID_SIZE
    states = np.arange(n_states, dtype=np.int64)
    x = np.eye(n_states, dtype=np.float32)[states]

    labels = []
    for s in states:
        labels.append(
            [
                float(next_state(s, 0) in HOLE_STATES),  # left_to_hole
                float(next_state(s, 1) in HOLE_STATES),  # down_to_hole
                float(next_state(s, 2) in HOLE_STATES),  # right_to_hole
                float(next_state(s, 3) in HOLE_STATES),  # up_to_hole
            ]
        )
    y = np.asarray(labels, dtype=np.float32)
    return x, y


def main():
    print("[1/4] Building supervised dataset...")
    x, y = build_supervised_dataset()
    print(f"      Dataset: X={x.shape}, y={y.shape}")

    print("[2/4] Training MLP sensor...")
    th.manual_seed(SEED)
    np.random.seed(SEED)

    model = FrozenLakeSensorMLP(
        in_dim=GRID_SIZE * GRID_SIZE,
        hidden_sizes=MLP_HIDDEN_SIZES,
        out_dim=4,
    )
    optimizer = th.optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = th.nn.BCEWithLogitsLoss()

    xb = th.tensor(x, dtype=th.float32)
    yb = th.tensor(y, dtype=th.float32)

    losses = []
    pbar = tqdm(range(MLP_EPOCHS), desc="Pretraining sensor", unit="epoch")
    for _ in pbar:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    print("[3/4] Saving checkpoint...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "grid_size": GRID_SIZE,
        "hidden_sizes": tuple(MLP_HIDDEN_SIZES),
        "out_dim": 4,
    }
    th.save(payload, CHECKPOINT_PATH)
    print(f"      Saved checkpoint: {CHECKPOINT_PATH}")

    print("[4/4] Saving pretraining curve...")
    hist = pd.DataFrame(
        {"epoch": np.arange(1, MLP_EPOCHS + 1), "bce_loss": np.asarray(losses)}
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(hist["epoch"], hist["bce_loss"], lw=2)
    ax.set_title("FrozenLake Sensor Pretraining Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "frozenlake_pretrain_loss.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"      Saved figure: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
