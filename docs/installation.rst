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

- ``stable-baselines3[extra]``
- ``sb3-contrib`` (required for TRPO support)

From PyPI
---------

Command: |cmd_pip_install|

.. code-block:: bash

   pip install clean-pls

This installs ``sb3-contrib`` via package dependencies.

For development
---------------

Command: |cmd_pip_editable|

.. code-block:: bash

   pip install -e .

This editable install also includes ``sb3-contrib`` from project dependencies.

If you created your environment before TRPO support was added, run:

.. code-block:: bash

   pip install -U sb3-contrib

Build docs dependencies
-----------------------

Command: |cmd_pip_docs|

.. code-block:: bash

   pip install -e .[docs]

Build documentation
-------------------

Command: |cmd_docs_build|

.. code-block:: bash

   make -C docs html
