"""Compare the previous sensor-only benchmark with a device-resident JAX pipeline.

The earlier benchmark timed only MLP sensor pretraining.  This script adds a
larger JAX workload in which the following remain on CPU only during setup:

* JAX device discovery,
* final scalar conversion,
* CSV and plot writing.

During each timed update, JAX handles the vectorized FrozenLake transitions,
sensor inference, policy/value networks, action sampling, shield masking,
advantage calculation, PPO-style losses, and Adam updates on one device.
Gymnasium is not stepped in this benchmark because its Python environment
interface is CPU-bound; the JAX transition function is equivalent for the
deterministic 4x4 FrozenLake map.

Run from repository root:
    python examples/frozenlake/benchmark_jax_gpu_pipeline.py

The script preserves the previous sensor-only rows in a combined CSV and
generates a comparison plot.  Use environment variables to shorten a smoke
run, for example:
    CLEANPLS_PIPELINE_UPDATES=10 python examples/frozenlake/benchmark_jax_gpu_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# Global Benchmark Settings
# =========================
UPDATES = int(os.getenv("CLEANPLS_PIPELINE_UPDATES", "100"))
WARMUP_UPDATES = int(os.getenv("CLEANPLS_PIPELINE_WARMUP", "5"))
SENSOR_EPOCHS = int(os.getenv("CLEANPLS_PIPELINE_SENSOR_EPOCHS", "20"))
NUM_ENVS = int(os.getenv("CLEANPLS_PIPELINE_ENVS", "256"))
ROLLOUT_STEPS = int(os.getenv("CLEANPLS_PIPELINE_ROLLOUT_STEPS", "16"))
SEED = 0
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("CLEANPLS_PIPELINE_OUTPUT_DIR", str(ROOT / "images")))
PREVIOUS_RESULTS_PATH = ROOT / "images" / "frozenlake_torch_vs_jax_benchmark.csv"
RESULTS_PATH = OUTPUT_DIR / "frozenlake_jax_gpu_pipeline_comparison.csv"
PLOT_PATH = OUTPUT_DIR / "frozenlake_jax_gpu_pipeline_comparison.png"
WORKER_PATH = Path(__file__).resolve()


def _jax_worker(device_name: str) -> dict:
    import jax

    requested_platform = "metal" if device_name == "metal" else "cpu"
    devices = [d for d in jax.devices() if d.platform.lower() == requested_platform]
    if not devices:
        return {
            "backend": "jax",
            "device": device_name,
            "status": "skipped",
            "reason": f"JAX {requested_platform} unavailable",
        }

    from pls.jax.pipeline import (
        FrozenLakeJaxPipelineConfig,
        run_pipeline_benchmark,
    )

    config = FrozenLakeJaxPipelineConfig(
        num_envs=NUM_ENVS,
        rollout_steps=ROLLOUT_STEPS,
    )
    return run_pipeline_benchmark(
        seed=SEED,
        updates=UPDATES,
        warmup_updates=WARMUP_UPDATES,
        sensor_epochs=SENSOR_EPOCHS,
        pipeline_config=config,
        device=devices[0],
    )


def worker_main(device: str):
    print(json.dumps(_jax_worker(device)))


def run_worker(device: str) -> dict:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "METAL" if device == "metal" else "cpu"
    command = [sys.executable, str(WORKER_PATH), "--worker", "--device", device]
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    json_lines = [
        line for line in completed.stdout.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    if json_lines:
        return json.loads(json_lines[-1])
    reason = (
        completed.stderr.strip().splitlines()[-1]
        if completed.stderr.strip()
        else "worker exited without a result"
    )
    return {"backend": "jax", "device": device, "status": "skipped", "reason": reason}


def load_previous_results() -> pd.DataFrame:
    """Load the earlier sensor-only results as a comparison workload."""
    if not PREVIOUS_RESULTS_PATH.exists():
        return pd.DataFrame()
    previous = pd.read_csv(PREVIOUS_RESULTS_PATH)
    previous = previous[previous["status"] == "ok"].copy()
    previous["workload"] = "sensor-only baseline"
    previous["metric_ms"] = previous["step_ms"]
    previous["metric_label"] = "ms per sensor-training step"
    return previous


def plot_results(frame: pd.DataFrame):
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return
    ok["label"] = ok["workload"] + "\n" + ok["backend"].str.upper() + " " + ok["device"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ok["label"], ok["metric_ms"])
    ax.set_ylabel("Milliseconds per measured unit")
    ax.set_title("FrozenLake: Previous Sensor Benchmark vs Device-Resident JAX Pipeline")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", choices=("cpu", "metal"))
    args = parser.parse_args()
    if args.worker:
        worker_main(args.device)
        return

    print("JAX GPU pipeline benchmark")
    print(
        f"updates={UPDATES}, warmup={WARMUP_UPDATES}, "
        f"sensor_epochs={SENSOR_EPOCHS}, envs={NUM_ENVS}, rollout_steps={ROLLOUT_STEPS}"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current = []
    for device in ("cpu", "metal"):
        print(f"\n[Run] JAX {device} full pipeline ...")
        result = run_worker(device)
        result["workload"] = "full JAX pipeline"
        result["metric_ms"] = result.get("update_ms")
        result["metric_label"] = "ms per PPO-style update"
        current.append(result)
        if result["status"] == "ok":
            print(
                f"      {result['update_ms']:.3f} ms/update, "
                f"{result['transitions_per_second']:.1f} transitions/s, "
                f"policy safety={result['mean_policy_safety']:.4f}"
            )
        else:
            print(f"      skipped: {result.get('reason', 'unknown reason')}")

    previous = load_previous_results()
    current_frame = pd.DataFrame(current)
    combined = pd.concat([previous, current_frame], ignore_index=True, sort=False)
    combined.to_csv(RESULTS_PATH, index=False)
    plot_results(combined)

    print(f"\nSaved comparison CSV: {RESULTS_PATH}")
    if not combined[combined["status"] == "ok"].empty:
        print(f"Saved comparison plot: {PLOT_PATH}")
    print("\nComparison:")
    columns = ["workload", "backend", "device", "status", "metric_ms", "metric_label"]
    print(combined[[column for column in columns if column in combined]].to_string(index=False))


if __name__ == "__main__":
    main()

