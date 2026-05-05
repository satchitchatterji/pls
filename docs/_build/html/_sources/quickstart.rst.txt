Quickstart
==========

This page gives fast entry points for the two main usage styles:

1. minimal single-file runs (FrozenLake),
2. registry/config workflow runs (LineWorld or FrozenLake plugin).

LineWorld Config Workflow
-------------------------

Train the included LineWorld config-driven example:

.. code-block:: bash

   python examples/train_a_policy/lineworld_rule_based/quickstart.py

Main example config:

- ``examples/train_a_policy/lineworld_rule_based/seed1/config.json``

LineWorld notebook:

- ``examples/lineworld_benchmark_curves.ipynb``

FrozenLake Minimal Scripts (No Registry/Config)
-----------------------------------------------

FrozenLake self-contained runs (no registry/config required):

- ``python examples/frozenlake_oracle.py``
- ``python examples/frozenlake_mlp.py``

FrozenLake notebook comparisons:

- ``examples/frozenlake/frozenlake_compare_oracle_vs_pretrained_mlp.ipynb``
- ``examples/frozenlake/frozenlake_dqn_sdqn_variant_matrix.ipynb``

FrozenLake Registry/Config Workflow
-----------------------------------

FrozenLake registry/config quickstart:

- ``python examples/train_a_policy/frozenlake_plugin/quickstart.py``
- config: ``examples/train_a_policy/frozenlake_plugin/seed1/config.json``
