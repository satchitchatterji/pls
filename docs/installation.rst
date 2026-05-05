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

From PyPI
---------

Command: |cmd_pip_install|

.. code-block:: bash

   pip install clean-pls

For development
---------------

Command: |cmd_pip_editable|

.. code-block:: bash

   pip install -e .

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
