from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pls.sensors.mlp import LineWorldSensorMLP


@dataclass
class TrainConfig:
    length: int = 7
    include_center: bool = True
    hidden_sizes: tuple[int, ...] = (32, 32)
    train_examples: int = 4000
    test_examples: int = 1000
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 0


def make_dataset(n: int, length: int, include_center: bool, seed: int):
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, length, size=(n,))
    obs = (pos / float(length - 1)).astype(np.float32)

    near_left = (pos <= 1).astype(np.float32)
    near_right = (pos >= length - 2).astype(np.float32)
    labels = [near_left, near_right]
    if include_center:
        at_center = (pos == (length // 2)).astype(np.float32)
        labels.append(at_center)

    x = th.tensor(obs).unsqueeze(1)
    y = th.tensor(np.stack(labels, axis=1))
    return TensorDataset(x, y)


def evaluate(model: nn.Module, loader: DataLoader, device: th.device):
    model.eval()
    total = 0.0
    n = 0
    correct = 0
    elems = 0
    loss_fn = nn.BCEWithLogitsLoss()
    with th.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
            pred = (th.sigmoid(logits) > 0.5).float()
            correct += (pred == yb).sum().item()
            elems += yb.numel()
    return total / max(n, 1), correct / max(elems, 1)


def main(output_path: str):
    cfg = TrainConfig()
    th.manual_seed(cfg.seed)

    train_ds = make_dataset(cfg.train_examples, cfg.length, cfg.include_center, cfg.seed)
    test_ds = make_dataset(cfg.test_examples, cfg.length, cfg.include_center, cfg.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    out_dim = 3 if cfg.include_center else 2
    model = LineWorldSensorMLP(hidden_sizes=cfg.hidden_sizes, out_dim=out_dim)
    device = th.device("cpu")
    model.to(device)

    opt = th.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_loss, test_acc = evaluate(model, test_loader, device)
            print(f"epoch={epoch+1:02d} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "hidden_sizes": cfg.hidden_sizes,
        "include_center": cfg.include_center,
        "length": cfg.length,
    }
    th.save(payload, output_path)
    final_loss, final_acc = evaluate(model, test_loader, device)
    print(f"saved checkpoint: {output_path}")
    print(f"final_test_loss={final_loss:.4f} final_test_acc={final_acc:.4f}")


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))
    ckpt_path = os.path.join(cwd, "lineworld_mlp_sensor.pt")
    main(ckpt_path)
