"""Benchmark the same FrozenLake MLP workload in Torch and JAX.

The benchmark compares:

* Torch CPU vs Torch MPS (Apple GPU),
* JAX CPU vs JAX Metal (Apple GPU, via ``jax-metal``).

Each backend trains the same 16 -> 16 -> 16 -> 4 sensor with the same
BCE-with-logits objective.  Backend runs are isolated in subprocesses because
an incompatible JAX Metal plugin can abort during runtime initialization on a
machine without a compatible GPU.  Such a backend is recorded as skipped.

Run from repository root:
    python examples/frozenlake/benchmark_torch_vs_jax.py

Optional controls:
    CLEANPLS_BENCHMARK_STEPS=1000 python examples/frozenlake/benchmark_torch_vs_jax.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# Global Benchmark Settings
# =========================
STEPS = int(os.getenv("CLEANPLS_BENCHMARK_STEPS", "500"))
WARMUP_STEPS = int(os.getenv("CLEANPLS_BENCHMARK_WARMUP", "20"))
BATCH_SIZE = int(os.getenv("CLEANPLS_BENCHMARK_BATCH", "4096"))
SEED = 0
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("CLEANPLS_BENCHMARK_OUTPUT_DIR", str(ROOT / "images")))
RESULTS_PATH = OUTPUT_DIR / "frozenlake_torch_vs_jax_benchmark.csv"
PLOT_PATH = OUTPUT_DIR / "frozenlake_torch_vs_jax_benchmark.png"
WORKER_PATH = Path(__file__).resolve()


def _dataset(batch_size: int):
    states = np.arange(16, dtype=np.int32)
    x = np.eye(16, dtype=np.float32)[states]
    holes = {5, 7, 11, 12}

    def next_state(state: int, action: int) -> int:
        row, col = divmod(state, 4)
        if action == 0:
            col, row = max(0, col - 1), row
        elif action == 1:
            col, row = col, min(3, row + 1)
        elif action == 2:
            col, row = min(3, col + 1), row
        else:
            col, row = col, max(0, row - 1)
        return row * 4 + col

    y = np.asarray(
        [[float(next_state(int(s), a) in holes) for a in range(4)] for s in states],
        dtype=np.float32,
    )
    repeats = int(np.ceil(batch_size / len(x)))
    return np.tile(x, (repeats, 1))[:batch_size], np.tile(y, (repeats, 1))[:batch_size]


def _torch_worker(device_name: str) -> dict:
    import torch
    import torch.nn as nn

    if device_name == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            return {"backend": "torch", "device": "mps", "status": "skipped", "reason": "Torch MPS unavailable"}
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(SEED)
    x_np, y_np = _dataset(BATCH_SIZE)
    x = torch.as_tensor(x_np, device=device)
    y = torch.as_tensor(y_np, device=device)
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    if device_name == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    if device_name == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "backend": "torch",
        "device": device_name,
        "status": "ok",
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "seconds": elapsed,
        "step_ms": elapsed / STEPS * 1000.0,
        "steps_per_second": STEPS / elapsed,
        "final_loss": float(loss.detach().cpu()),
    }


def _jax_worker(device_name: str) -> dict:
    import jax
    import jax.numpy as jnp

    requested_platform = "metal" if device_name == "metal" else "cpu"
    devices = [d for d in jax.devices() if d.platform.lower() == requested_platform]
    if not devices:
        return {"backend": "jax", "device": device_name, "status": "skipped", "reason": f"JAX {requested_platform} unavailable"}
    device = devices[0]

    from pls.jax.frozenlake import FrozenLakeSensorConfig

    # Use the same optimizer/model workload as the JAX implementation, with a
    # large repeated batch to make accelerator timing meaningful.
    x_np, y_np = _dataset(BATCH_SIZE)
    config = FrozenLakeSensorConfig(grid_size=4, hidden_sizes=(16, 16), out_dim=4)
    key = jax.random.PRNGKey(SEED)
    dims = (16, 16, 16, 4)
    keys = jax.random.split(key, 3)
    params = []
    for subkey, in_dim, out_dim in zip(keys, dims[:-1], dims[1:]):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        params.append({"w": jax.random.uniform(subkey, (in_dim, out_dim), minval=-limit, maxval=limit), "b": jnp.zeros((out_dim,))})
    x = jax.device_put(jnp.asarray(x_np), device)
    y = jax.device_put(jnp.asarray(y_np), device)

    def apply(current_params, features):
        hidden = features
        for layer in current_params[:-1]:
            hidden = jax.nn.relu(hidden @ layer["w"] + layer["b"])
        layer = current_params[-1]
        return hidden @ layer["w"] + layer["b"]

    def loss_fn(current_params):
        logits = apply(current_params, x)
        return jnp.mean(jnp.maximum(logits, 0.0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))

    opt_state = {
        "m": [{"w": jnp.zeros_like(p["w"]), "b": jnp.zeros_like(p["b"])} for p in params],
        "v": [{"w": jnp.zeros_like(p["w"]), "b": jnp.zeros_like(p["b"])} for p in params],
        "step": 0,
    }

    def step(current_params, state):
        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        step_index = state["step"] + 1
        next_params, next_m, next_v = [], [], []
        for p, g, m, v in zip(current_params, grads, state["m"], state["v"]):
            p_new, m_new, v_new = {}, {}, {}
            for name in ("w", "b"):
                m_value = 0.9 * m[name] + 0.1 * g[name]
                v_value = 0.999 * v[name] + 0.001 * jnp.square(g[name])
                m_hat = m_value / (1 - 0.9**step_index)
                v_hat = v_value / (1 - 0.999**step_index)
                p_new[name] = p[name] - 1e-2 * m_hat / (jnp.sqrt(v_hat) + 1e-8)
                m_new[name] = m_value
                v_new[name] = v_value
            next_params.append(p_new)
            next_m.append(m_new)
            next_v.append(v_new)
        return next_params, {"m": next_m, "v": next_v, "step": step_index}, loss

    step = jax.jit(step)
    with jax.default_device(device):
        for _ in range(WARMUP_STEPS):
            params, opt_state, loss = step(params, opt_state)
        jax.block_until_ready(params)
        start = time.perf_counter()
        for _ in range(STEPS):
            params, opt_state, loss = step(params, opt_state)
        jax.block_until_ready(params)
        elapsed = time.perf_counter() - start
    return {
        "backend": "jax",
        "device": device_name,
        "status": "ok",
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "seconds": elapsed,
        "step_ms": elapsed / STEPS * 1000.0,
        "steps_per_second": STEPS / elapsed,
        "final_loss": float(loss),
        "jax_device": str(device),
        "jax_version": jax.__version__,
        "jax_config": config.hidden_sizes,
    }


def worker_main(backend: str, device: str):
    result = _torch_worker(device) if backend == "torch" else _jax_worker(device)
    print(json.dumps(result))


def run_worker(backend: str, device: str) -> dict:
    env = os.environ.copy()
    if backend == "jax":
        env["JAX_PLATFORMS"] = "METAL" if device == "metal" else "cpu"
    command = [sys.executable, str(WORKER_PATH), "--worker", "--backend", backend, "--device", device]
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    json_lines = [line for line in completed.stdout.splitlines() if line.startswith("{") and line.endswith("}")]
    if json_lines:
        return json.loads(json_lines[-1])
    reason = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "worker exited without a result"
    return {"backend": backend, "device": device, "status": "skipped", "reason": reason}


def plot_results(results: pd.DataFrame):
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return
    ok["label"] = ok["backend"].str.upper() + " " + ok["device"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ok["label"], ok["step_ms"])
    ax.set_ylabel("Milliseconds per training step")
    ax.set_title("FrozenLake Sensor Training: Torch vs JAX")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=("torch", "jax"))
    parser.add_argument("--device", choices=("cpu", "mps", "metal"))
    args = parser.parse_args()
    if args.worker:
        worker_main(args.backend, args.device)
        return

    print("Torch/JAX FrozenLake sensor benchmark")
    print(f"steps={STEPS}, warmup={WARMUP_STEPS}, batch_size={BATCH_SIZE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for backend, device in (("torch", "cpu"), ("torch", "mps"), ("jax", "cpu"), ("jax", "metal")):
        print(f"\n[Run] {backend.upper()} {device} ...")
        result = run_worker(backend, device)
        results.append(result)
        if result["status"] == "ok":
            print(f"      {result['step_ms']:.3f} ms/step, final loss={result['final_loss']:.6f}")
        else:
            print(f"      skipped: {result.get('reason', 'unknown reason')}")

    frame = pd.DataFrame(results)
    frame.to_csv(RESULTS_PATH, index=False)
    plot_results(frame)
    print(f"\nSaved metrics: {RESULTS_PATH}")
    if not frame[frame["status"] == "ok"].empty:
        print(f"Saved plot: {PLOT_PATH}")
    print("\nResults:")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
