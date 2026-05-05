Tutorial 2: FrozenLake Pretrained MLP Sensor (No Registry, No Config)
======================================================================

This tutorial uses the fully self-contained script:

- ``examples/frozenlake_mlp.py``

Goal
----

Run a complete SPPO experiment where the sensor is a pretrained MLP, still
without registry/config plumbing.

Run
---

.. code-block:: bash

   python examples/frozenlake_mlp.py

What it does
------------

1. Creates ``FrozenLake-v1`` (4x4, slippery).
2. Generates supervised labels from map geometry:
   - for each state and action, label whether the next cell is a hole.
3. Pretrains an MLP sensor on those labels.
4. Wraps the pretrained model behind ``predict(obs, info=None)``.
5. Uses the same in-file ProbLog shield structure as the oracle tutorial.
6. Trains ``PPO_shielded`` with the pretrained sensor.
7. Plots and saves:
   - MLP pretraining BCE loss,
   - RL reward curve,
   - RL failure-rate curve.

Saved images
------------

- ``examples/images/frozenlake/frozenlake_mlp_curves.png``
- ``examples/images/frozenlake/frozenlake_mlp_pretrain_only.png`` (fallback path)

Why use this tutorial
---------------------

Use this path when you want a compact research script to compare:

- idealized oracle sensing vs
- learned sensing quality

without introducing config/registry complexity.

Related notebooks
-----------------

- ``examples/frozenlake/frozenlake_compare_oracle_vs_pretrained_mlp.ipynb``
- ``examples/frozenlake/frozenlake_dqn_sdqn_variant_matrix.ipynb``
