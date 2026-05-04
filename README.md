# CleanPLS

CleanPLS is a stripped-down implementation of probabilistic logic shields for reinforcement learning.

Install from PyPI (package name): `clean-pls`  
Import in Python: `import pls`

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
- Pretrained MLP sensor model registry:
  - `pretrained_lineworld_mlp_v1`
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

## Install

Published package install:

```bash
pip install clean-pls
```

Local development install:

```bash
pip install -e .
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

## Pretrain an MLP Sensor (LineWorld)

Train and save the LineWorld MLP sensor wrapper:

```bash
python examples/pretrain_sensors_lineworld/pretrain_mlp_sensor.py
```

Checkpoint output:

- `examples/pretrain_sensors_lineworld/lineworld_mlp_sensor.pt`

Use pretrained sensor in training config:

- `examples/train_a_policy/lineworld_mlp_sensor/seed1/config.json`

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

## Build and Publish

Build artifacts:

```bash
python -m build
```

Validate package metadata:

```bash
twine check dist/*
```

Upload to PyPI:

```bash
twine upload dist/*
```

Recommended first release dry run (TestPyPI):

```bash
twine upload --repository testpypi dist/*
```

## Documentation (Sphinx)

Install docs dependencies:

```bash
pip install -e .[docs]
```

Build HTML docs:

```bash
make -C docs html
```

Open:

- `docs/_build/html/index.html`

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

Example pretrained MLP sensor block:

```json
"sensor": {
  "type": "mlp_pretrained",
  "name": "pretrained_lineworld_mlp_v1",
  "params": {
    "checkpoint_path": "../../../pretrain_sensors_lineworld/lineworld_mlp_sensor.pt",
    "hidden_sizes": [32, 32],
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
