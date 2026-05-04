# CleanPLS

CleanPLS is a stripped-down implementation of probabilistic logic shields for reinforcement learning.

This repo focuses on:
- Gymnasium-compatible environments
- Shielded and unshielded policy learning with Stable-Baselines3
- Easy experimentation on a minimal safety environment (`LineWorldSafety-v0`)
- A generic non-trainable sensor interface that can later be replaced by trainable models (for example, CNN-based sensors)

## What Is Included

- `LineWorldSafety-v0`: a tiny 1D safety test environment
- Shielded algorithms:
  - `PPO_shielded`
  - `A2C_shielded`
- Unshielded baselines via SB3:
  - `PPO`
  - `A2C`
- Rule-based sensor model registry:
  - `rule_based_lineworld_v1`
- ProbLog shield program for LineWorld:
  - `examples/train_a_policy/data/lineworld_safety.pl`
- Benchmark notebook with reward/safety curves:
  - `examples/lineworld_benchmark_curves.ipynb`
- Basic tests for env/sensor/shield integration:
  - `tests/`

## Repository Layout

- `pls/envs/` Gymnasium environments and environment registration
- `pls/sensors/` sensor abstractions and registry
- `pls/shields/` probabilistic logic shield implementation
- `pls/algorithms/` shielded algorithm implementations and train/eval helpers
- `pls/workflows/` training/evaluation entry points
- `examples/` runnable configs, shield programs, and notebook
- `tests/` smoke/integration tests

## Requirements

- Python 3.11+
- Conda (recommended)

## Setup (Conda)

Use the provided environment file:

```bash
conda env create -f environment.yml
conda activate CleanPLS
python -m ipykernel install --user --name CleanPLS --display-name "Python (CleanPLS)"
```

## Quick Start: Train LineWorld Agent

Run the included quickstart:

```bash
python examples/train_a_policy/lineworld_rule_based/quickstart.py
```

Config used:

- `examples/train_a_policy/lineworld_rule_based/seed1/config.json`

That config uses:
- `env: LineWorldSafety-v0`
- `sensor: rule_based_lineworld_v1`
- shield enabled with `lineworld_safety.pl`

## Benchmark Notebook (Reward + Safety Curves)

Open and run:

- `examples/lineworld_benchmark_curves.ipynb`

The notebook compares:
- A2C
- PPO
- Shielded A2C
- Shielded PPO

and plots:
- training reward curves
- unsafe termination rate curves

## Run Tests

```bash
pytest -q tests/test_lineworld_and_sensors.py tests/test_shield_lineworld.py
```

## Configuration Notes

Current workflow supports:
- `algorithm: "ppo"` or `"a2c"`
- Gymnasium environment IDs
- optional `sensor` block with registry-backed construction

Example sensor block:

```json
"sensor": {
  "type": "rule_based",
  "name": "rule_based_lineworld_v1",
  "params": {
    "include_center": true
  }
}
```

## Current Scope and Limitations

- Designed for discrete action spaces in the shielded path
- LineWorld is a minimal validation environment, not a benchmark suite
- Sensor model in this repo is rule-based; trainable sensor models are planned as next extension

## License

See `LICENSE`.
