Tutorial: Research Workflow for a New Environment
=================================================

This tutorial is written for research iteration, not framework internals.
The objective is to move from hypothesis to reproducible results with minimal code edits.

Research Framing First
----------------------

Before touching code, write three lines:

1. **Task hypothesis**: what behavior should the shield improve?
2. **Sensor hypothesis**: what latent facts does safety depend on?
3. **Evaluation hypothesis**: what safety/return trade-off do you expect?

Example:

- Task: avoid unsafe boundary actions in a stochastic control environment.
- Sensors: near-boundary indicators and one uncertainty indicator.
- Expectation: shield lowers unsafe rate with small return penalty.

Minimum Integration Surface
---------------------------

To plug in a new study, you need only:

- a Gymnasium environment id,
- a sensor wrapper (rule-based or pretrained),
- a ProbLog shield program,
- a config file.

Core workflow code should not be edited for each new experiment.

Step 1: Register the Environment Runtime Once
----------------------------------------------

CleanPLS uses registries so new environments are attached once and reused by configs.

Runtime registries:

- ``ALGO_REGISTRY``: algorithm id -> class
- ``ENV_REGISTRY``: env id -> runtime bundle resolver

Location:

- ``pls/workflows/registry.py``

Minimal resolver pattern:

.. code-block:: python

   from pls.workflows.registry import register_env_runtime, RuntimeBundle

   def my_env_resolver(algorithm_cls):
       return RuntimeBundle(
           model_cls=algorithm_cls,
           get_sensor_value_ground_truth=my_ground_truth_fn,
           custom_callback_class=None,
           monitor_cls=None,
           features_extractor_cls=None,
           observation_net_cls=None,
       )

   register_env_runtime("MySafetyEnv-v0", my_env_resolver)

Research note:

- Treat resolver code as static infrastructure.
- Treat configs as your experiment interface.

Step 2: Define Sensors as Experimental Variables
------------------------------------------------

Sensor wrappers should be comparable experimental conditions.

Contract:

- ``SensorModel.predict(obs, info=None) -> tensor[batch, num_sensors]``
- values in ``[0, 1]``
- sensor column order must match shield ``sensor_value(i)`` order.

Two common conditions:

- ``rule_based`` for oracle/procedural sensors,
- ``mlp_pretrained`` for learned sensors.

Registry location:

- ``pls/sensors/registry.py``

Sensor Pretraining (Recommended for Learned Sensor Studies)
-----------------------------------------------------------

If your study compares oracle/rule-based sensors against learned sensors,
include an explicit pretraining stage before policy training.

Reference implementation in this repo:

- Script: ``examples/pretrain_sensors_lineworld/pretrain_mlp_sensor.py``
- Output checkpoint: ``examples/pretrain_sensors_lineworld/lineworld_mlp_sensor.pt``

Run pretraining:

.. code-block:: bash

   python examples/pretrain_sensors_lineworld/pretrain_mlp_sensor.py

Then use the checkpoint in config:

.. code-block:: json

   "sensor": {
     "type": "mlp_pretrained",
     "name": "pretrained_lineworld_mlp_v1",
     "params": {
       "checkpoint_path": "../../../pretrain_sensors_lineworld/lineworld_mlp_sensor.pt",
       "hidden_sizes": [32, 32],
       "include_center": true
     }
   }

Research reporting guidance:

- Report sensor pretraining data generation assumptions.
- Report sensor validation metrics (e.g., per-label accuracy/F1).
- Keep sensor architecture fixed when comparing shield effects.
- Separate two ablations:
  - **Shield effect**: fixed sensor, shield off vs on.
  - **Sensor effect**: fixed shield, rule-based vs pretrained sensor.

Step 3: Write Shield Program as an Assumption Sheet
---------------------------------------------------

Think of the ``.pl`` file as formalized safety assumptions.

Include:

1. action declarations,
2. sensor declarations,
3. unsafe clauses,
4. ``safe_next`` definition.

Starter template:

- ``examples/templates/shield_program_starter.pl``

Critical consistency checks:

- config ``num_actions`` equals number of declared actions,
- config ``num_sensors`` equals number of sensor declarations,
- sensor wrapper output ordering matches declarations.

Step 4: Start from a Reproducible Config Template
-------------------------------------------------

Use:

- ``examples/templates/experiment_config_template.json``
- ``examples/templates/experiment_config_template.md``

Required blocks for shielded runs:

- ``env``, ``algorithm``, ``policy_params``
- ``shield_params``
- ``policy_safety_params``

Optional block:

- ``sensor``

Path semantics:

- ``shield_program`` and ``sensor.params.checkpoint_path`` may be relative.
- They are resolved against the config file directory.

Step 5: Run Core Experimental Matrix
------------------------------------

At minimum, run these ablations:

1. no shield,
2. shield + rule-based sensor,
3. shield + pretrained sensor.

Then sweep:

- ``alpha`` (safety coefficient),
- random seeds.

Helper script for reproducible matrix + CSV export:

.. code-block:: bash

   python examples/research_compare.py \
     --configs examples/train_a_policy/lineworld_rule_based/seed1/config.json \
               examples/train_a_policy/lineworld_mlp_sensor/seed1/config.json \
     --seeds 0 1 2 \
     --timesteps 2000 \
     --eval-episodes 20 \
     --out results/comparison.csv

Recommended report columns:

- mean reward,
- unsafe-rate metric,
- seed,
- algorithm,
- sensor condition,
- shield condition.

Preflight Validation (Fail Fast)
--------------------------------

Before training starts, CleanPLS validates:

- shield file exists,
- action-space size matches ``num_actions``,
- sensor output width matches ``num_sensors`` (if sensor wrapper is configured),
- checkpoint path exists for pretrained sensors.

This is designed to reduce failed overnight experiments.

Ablation Cookbook
-----------------

Use these standard experiment families:

- **Safety effect**: same policy, shield off vs on.
- **Sensor effect**: shielded policy, rule-based vs pretrained MLP.
- **Penalty effect**: shielded policy, alpha sweep (e.g., 0.0/0.05/0.1/0.2).
- **Stability**: fixed setup, multi-seed variability.

Interpretation guidance:

- If unsafe rate drops and reward is stable, shield assumptions are likely aligned.
- If unsafe rate stays high, check sensor quality and sensor-shield semantic mismatch.
- If reward collapses, inspect over-constrained unsafe rules or alpha too large.

Troubleshooting by Research Failure Mode
----------------------------------------

**Unexpected unsafe behavior persists**

- Verify sensor ordering against ``sensor_value(i)`` declarations.
- Check thresholding/calibration for learned sensors.

**Immediate runtime error**

- Most often path resolution or dimension mismatch.
- Start from the template config and change one block at a time.

**High variance across seeds**

- Increase timesteps and report confidence intervals, not single-seed results.
- Keep environment stochasticity explicit in config and writeup.

Practical Checklist for New Task Onboarding
--------------------------------------------

- Environment id is registered in ``ENV_REGISTRY``.
- Sensor wrapper is registered in ``SENSOR_REGISTRY``.
- Shield program and sensor ordering are aligned.
- Template config runs end-to-end with one seed.
- Ablation matrix runs across multiple seeds and exports CSV.
- Study notes record task/sensor/safety hypotheses and expected trade-offs.
