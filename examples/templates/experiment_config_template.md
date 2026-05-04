# Experiment Config Template Guide

This template is meant for research iteration speed.

## Required blocks
- `env`, `algorithm`, `policy_params`
- `shield_params`, `policy_safety_params` (if shielded run)

## Sensor choices
- Rule-based:
  - `sensor.type = "rule_based"`
  - `sensor.name = "..."` registered in `pls.sensors.registry`
- Pretrained MLP:
  - `sensor.type = "mlp_pretrained"`
  - `sensor.params.checkpoint_path` relative to config directory or absolute

## Path semantics
- `shield_program` and `checkpoint_path` are resolved relative to the config file directory.

## Fast ablations
- No shield: set `shield_params` and `policy_safety_params` to `null`, and `alpha=0`.
- Shield on: restore both shield blocks and tune `alpha`.
- Sensor study: swap `sensor` block between rule-based and pretrained.
