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

Suggested next steps
--------------------

1. Duplicate ``examples/train_a_policy/frozenlake_plugin/seed1/config.json`` for additional seeds/ablations.
2. Compare oracle/rule-based and pretrained sensor conditions.
3. Sweep shield coefficient ``alpha`` and report reward-safety trade-offs.
