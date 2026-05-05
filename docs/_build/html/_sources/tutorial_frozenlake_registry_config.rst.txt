Tutorial 3: FrozenLake with Registry + Config Workflow
======================================================

This tutorial uses the repository's registry/config resources for a reusable,
multi-experiment workflow.

Key files
---------

- Registration entrypoint:
  - ``examples/train_a_policy/frozenlake_plugin/research_registration.py``
- Workflow quickstart:
  - ``examples/train_a_policy/frozenlake_plugin/quickstart.py``
- Experiment config:
  - ``examples/train_a_policy/frozenlake_plugin/seed1/config.json``

Goal
----

Run FrozenLake by registering runtime/sensor once, then driving experiments
through config files.

Current status of the bundled config
------------------------------------

The shipped example config currently demonstrates an ``a2c``-based run.
You can reuse the same pattern for:

- ``ppo``
- ``dqn`` (shielded DQN / PLTD)
- ``dqn_vanilla``

by editing the algorithm and policy parameter blocks in your copied config.

Run
---

.. code-block:: bash

   python examples/train_a_policy/frozenlake_plugin/quickstart.py

What this path gives you
------------------------

1. Clear separation of responsibilities:
   - runtime wiring in registration file,
   - experiment parameters in config.
2. Easy reproducibility and parameter sweeps.
3. Direct compatibility with workflow utilities such as:
   - ``pls.workflows.execute_workflow.train``
   - ``examples/research_compare.py``

Recommended config fields for DQN studies
-----------------------------------------

When using ``algorithm: "dqn"``, additionally consider:

- ``exploration_policy``: ``"epsilon_greedy"`` or ``"softmax"``
- ``differentiable_exploration``: ``true``/``false``
- ``pltd_mode``: ``"off_policy"`` or ``"on_policy"``
- ``softmax_temperature``: float value for softmax policy shaping

Suggested next steps
--------------------

1. Duplicate ``examples/train_a_policy/frozenlake_plugin/seed1/config.json`` for additional seeds/ablations.
2. Compare oracle/rule-based and pretrained sensor conditions.
3. Sweep shield coefficient ``alpha`` and report reward-safety trade-offs.

Related notebooks
-----------------

- ``examples/frozenlake/frozenlake_compare_oracle_vs_pretrained_mlp.ipynb``
- ``examples/frozenlake/frozenlake_dqn_sdqn_variant_matrix.ipynb``
