Tutorial 2: FrozenLake Pretrained MLP Sensor (No Registry, No Config)
======================================================================

This tutorial uses two fully self-contained scripts:

- ``examples/frozenlake/frozenlake_pretrain.py``
- ``examples/frozenlake/frozenlake_pretrained.py``

Goal
----

Run a complete SPPO experiment where the sensor is a pretrained MLP, still
without registry/config plumbing.

Run
---

Step 1: pretrain and save a loadable sensor checkpoint:

.. code-block:: bash

   python examples/frozenlake/frozenlake_pretrain.py

Step 2: train SPPO using that saved checkpoint:

.. code-block:: bash

   python examples/frozenlake/frozenlake_pretrained.py

What it does
------------

1. ``frozenlake_pretrain.py`` creates a supervised dataset from map geometry:
   - for each state and action, label whether the next cell is a hole.
2. ``frozenlake_pretrain.py`` trains an MLP sensor and saves ``.pt`` weights.
3. ``frozenlake_pretrained.py`` loads that checkpoint into a sensor wrapper with
   ``predict(obs, info=None)``.
4. ``frozenlake_pretrained.py`` uses an in-file ProbLog shield and trains
   ``PPO_shielded``.
5. Plots and saves:
   - MLP pretraining BCE loss,
   - RL reward curve,
   - RL failure-rate proxy curve.

Saved artifacts
---------------

- ``examples/frozenlake/frozenlake_sensor_mlp.pt``
- ``examples/frozenlake/images/frozenlake_pretrain_loss.png``
- ``examples/frozenlake/images/frozenlake_pretrained_curves.png``

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
