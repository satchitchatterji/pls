Installation
============

Requirements
------------

- |python_min|

Recommended (Conda)
-------------------

Use the repository environment file:
Environment name: |conda_env_name|

.. code-block:: bash

   conda env create -f environment.yml
   conda activate CleanPLS

The bundled ``environment.yml`` includes:

- ``stable-baselines3``
- ``sb3-contrib`` (required for TRPO support)

Source-tree workflow
--------------------

CleanPLS is used directly from the repository in the documented research
workflow. After activating the Conda environment, run examples and tests from
the repository root. Python resolves the local ``pls`` package from the source
tree, so no separate distribution installation is required.

The environment specification includes the documentation dependencies.

Build documentation
-------------------

Command: |cmd_docs_build|

.. code-block:: bash

   make -C docs html
