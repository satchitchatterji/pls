# Algorithms TODO (Priority Ordered)

## P0 (Highest Priority)

1. Replace tentative continuous-action softmax relaxation with a principled continuous shielding formulation.
2. Unify workflow behavior so shielded continuous-action runs are either fully supported end-to-end or fail-fast with explicit guidance.
3. Add strict runtime assertions/logging for when safety loss is skipped due to action-dimension and `num_actions` mismatch.

## P1 (High Priority)

4. Add algorithm-specific tests for `SAC_shielded`, `TD3_shielded`, `DDPG_shielded`, and `TRPO_shielded` that verify:
   - safety loss is non-zero when expected,
   - loss is skipped only under documented conditions,
   - training remains numerically stable.
5. Add regression tests for DQN-family variants (`DQN_shielded`, `DoubleDQN_shielded`, `Rainbow_shielded`) under shared shield configs.
6. Standardize and document per-algorithm safety diagnostics (`P_safe`, intervention rate, unsafe-mass removed, safety loss scale).

## P2 (Medium Priority)

7. Upgrade `Rainbow_shielded` from Rainbow-lite to full Rainbow components (prioritized replay, n-step returns, noisy nets, distributional head).
8. Add reproducible benchmark scripts/notebooks comparing base vs `sbase(oracle)` vs `sbase(mlp)` across supported algorithm families.
9. Harmonize default hyperparameters and expose a single config schema table for all algorithms (required/optional keys).

## P3 (Lower Priority)

10. Improve API consistency across wrappers (constructor signatures, defaults, warning messages, logger keys).
10.1 Standardize constructor signatures across all shielded wrappers to a common order and names:
`alpha`, `shield_params`, `policy_safety_params`, `config_folder`, `get_sensor_value_ground_truth`.
10.2 Remove mutable-default patterns (for example `policy_safety_params={}`) and use `None` + internal normalization everywhere.
10.3 Normalize shielding parameter plumbing so every wrapper either:
- accepts both `shield_params` and `policy_safety_params` at top-level, or
- uses one documented canonical pathway (no mixed policy_kwargs-only exceptions).
10.4 Align behavior when `policy_safety_params` is empty:
- avoid constructing empty `Shield` objects in some wrappers while using `None` checks in others.
10.5 Unify warning policy:
- continuous-action fallback warnings should be emitted consistently in all relevant wrappers (including TRPO/DQN paths where applicable),
- wording should follow one template with class name + exact disable/fallback condition.
10.6 Add one shared logger-key schema and migrate wrappers to it:
- standard keys for actor term (`train/actor_loss`),
- critic/value term (`train/critic_loss` or `train/value_loss` with clear mapping),
- policy objective (`train/policy_loss` or `train/policy_objective`, pick one convention per family and document it),
- TD term (`train/td_loss` for value-based methods),
- safety term (`train/safety_loss`),
- total objective (`train/loss`) when available.
10.7 Ensure safety logging semantics are consistent:
- always record `train/safety_loss` (including explicit `0.0` when inactive),
- avoid conditional omission that breaks downstream dashboards.
10.8 Standardize naming/typing for internal shield handles:
- use one spelling (e.g., `policy_safety_calculator`) instead of mixed variants like `...calculater`.
10.9 Add a wrapper-level API contract table in docs mapping each algorithm to:
- supported action spaces,
- shield mode (predict-time, train-time, both),
- expected logger keys.
10.10 Add compatibility tests that instantiate each wrapper with identical kwarg blocks and verify no unexpected `TypeError`/ignored-arg behavior.
11. Add lightweight calibration checks for sensor uncertainty impact on shielding quality.
12. Add optional ablation helpers for alpha sweeps and seed sweeps with standardized CSV outputs.
