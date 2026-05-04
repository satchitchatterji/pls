from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import tempfile
from pathlib import Path

from pls.workflows.execute_workflow import evaluate, train


def _abspath_from_cfg(cfg_dir: Path, maybe_path: str) -> str:
    p = Path(maybe_path)
    return str(p if p.is_absolute() else (cfg_dir / p).resolve())


def _prepare_config_for_run(base_cfg: dict, cfg_dir: Path, seed: int, total_timesteps: int | None):
    cfg = copy.deepcopy(base_cfg)
    cfg["policy_params"]["seed"] = seed
    if total_timesteps is not None:
        cfg["policy_params"]["total_timesteps"] = total_timesteps

    for key in ("shield_params", "policy_safety_params"):
        if isinstance(cfg.get(key), dict) and "shield_program" in cfg[key]:
            cfg[key]["shield_program"] = _abspath_from_cfg(cfg_dir, cfg[key]["shield_program"])

    sensor = cfg.get("sensor")
    if isinstance(sensor, dict):
        params = sensor.get("params", {})
        if "checkpoint_path" in params:
            params["checkpoint_path"] = _abspath_from_cfg(cfg_dir, params["checkpoint_path"])

    return cfg


def run_one_config(config_path: Path, seeds: list[int], total_timesteps: int | None, n_eval_episodes: int):
    cfg_dir = config_path.parent
    with open(config_path) as f:
        base_cfg = json.load(f)

    rows = []
    for seed in seeds:
        cfg = _prepare_config_for_run(base_cfg, cfg_dir, seed, total_timesteps)
        run_dir = Path(tempfile.mkdtemp(prefix="cleanpls_cmp_", dir="/private/tmp"))
        run_cfg = run_dir / "config.json"
        run_cfg.write_text(json.dumps(cfg))

        train(str(run_cfg))
        rewards, _ = evaluate(str(run_cfg), model_at_step="end", n_test_episodes=n_eval_episodes)
        mean_reward = sum(rewards) / len(rewards) if rewards else float("nan")
        unsafe_rate = sum(1 for r in rewards if r < 0) / len(rewards) if rewards else float("nan")

        rows.append(
            {
                "config": str(config_path),
                "seed": seed,
                "algorithm": cfg.get("algorithm", cfg.get("base_policy", "ppo")),
                "timesteps": cfg["policy_params"]["total_timesteps"],
                "n_eval_episodes": n_eval_episodes,
                "mean_reward": mean_reward,
                "unsafe_proxy_rate": unsafe_rate,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run reproducible CleanPLS comparison matrix and export CSV")
    parser.add_argument("--configs", nargs="+", required=True, help="Config JSON files")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps for all runs")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--out", default="comparison_results.csv", help="CSV output path")
    args = parser.parse_args()

    all_rows = []
    for cfg in args.configs:
        all_rows.extend(run_one_config(Path(cfg).resolve(), args.seeds, args.timesteps, args.eval_episodes))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
