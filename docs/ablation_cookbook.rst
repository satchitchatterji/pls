Ablation Cookbook
=================

This cookbook provides a default evaluation protocol for CleanPLS studies.

Core Matrix
-----------

For each environment, run:

- Baseline: unshielded policy
- Shield + rule-based sensor
- Shield + pretrained MLP sensor

Across:

- seeds: at least 3
- alpha: at least 3 values

Suggested alpha grid
--------------------

- ``0.0`` (control)
- ``0.05``
- ``0.1``
- ``0.2``

Suggested output table
----------------------

Columns:

- environment
- algorithm
- shield_condition
- sensor_condition
- alpha
- seed
- mean_reward
- unsafe_rate

Run helper
----------

Use:

.. code-block:: bash

   python examples/research_compare.py \
     --configs <config_a.json> <config_b.json> ... \
     --seeds 0 1 2 \
     --timesteps 5000 \
     --eval-episodes 20 \
     --out results/comparison.csv

Recommended plots
-----------------

- reward vs alpha
- unsafe rate vs alpha
- reward vs unsafe-rate Pareto scatter
- per-seed boxplots for stability

Interpretation patterns
-----------------------

- lower unsafe rate + stable reward: shield assumptions likely useful
- lower unsafe rate + large reward loss: over-constrained rules or high alpha
- no safety gain: sensor semantics or shield clauses likely misaligned
