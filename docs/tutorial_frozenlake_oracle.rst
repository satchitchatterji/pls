Tutorial 1: FrozenLake Oracle (No Registry, No Config)
======================================================

This tutorial uses the fully self-contained script:

- ``examples/frozenlake_oracle.py``

Goal
----

Run a complete shielded PPO (SPPO) experiment on ``FrozenLake-v1`` without:

- registry registration code,
- JSON config files.

Everything lives in one file:

- environment creation,
- sensor wrapper,
- ProbLog shield program (as a Python string),
- SPPO training,
- plotting and image export.

Run
---

.. code-block:: bash

   python examples/frozenlake_oracle.py

What it does
------------

1. Creates ``FrozenLake-v1`` (4x4, slippery).
2. Uses an oracle sensor that outputs action-to-hole risk signals:
   - ``left_to_hole``, ``down_to_hole``, ``right_to_hole``, ``up_to_hole``.
3. Defines a shield from an in-file ProbLog string.
4. Trains ``PPO_shielded`` end-to-end.
5. Plots reward and failure-rate curves.
6. Saves figures to:
   - ``examples/images/frozenlake/frozenlake_oracle_curves.png``

Why use this tutorial
---------------------

Use this path when you want the fastest possible experiment loop and do not
need framework-level abstraction yet.

