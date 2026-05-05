Minimal Experiment Checklist (Generic Gymnasium Env)
====================================================

Use this checklist before running a new safety experiment in CleanPLS-style workflows.

1. Environment
--------------

- Gymnasium env id/constructor runs with valid ``reset``/``step`` behavior.
- Action space is discrete (current shielded path assumption).
- Observation format and shape are known.

2. Safety Objective
-------------------

- Unsafe condition is explicitly defined in environment terms.
- Safety metric is selected:
  - unsafe-event rate, or
  - failure/cost rate, or
  - cumulative safety cost.

3. Sensor Wrapper
-----------------

- Safety-relevant sensor variables are chosen.
- Sensor index ordering is fixed and documented.
- Wrapper implements:
  - ``predict(obs, info=None) -> tensor[batch, num_sensors]`` with values in ``[0,1]``.

4. Shield Program (ProbLog)
---------------------------

- Action declarations are present.
- Sensor declarations are present.
- Unsafe clauses are present.
- ``safe_next :- \+ unsafe_next.`` is present.
- ``num_actions`` and ``num_sensors`` match code/config.

5. Policy + Shield Wiring
-------------------------

- Shield receives expected observations and sensor outputs.
- Shield params are complete:
  - ``num_sensors``
  - ``num_actions``
  - ``program_str`` or ``shield_program`` path
  - sensor model/wrapper handle
- If PLTD is used, set:
  - ``alpha``
  - ``pltd_mode`` (``off_policy`` or ``on_policy``)
  - differentiable path flag.

6. Logging and Instrumentation
------------------------------

- Per-episode reward is logged.
- Per-episode safety/failure/cost signal is logged.
- Timestep is logged.
- Progress bar and seed are visible in run output.

7. Baselines and Comparisons
----------------------------

- Run unshielded baseline.
- Run shielded oracle/rule-based condition.
- If learned sensor is used, run pretrained-sensor shielded condition.

8. Plots and Artifacts
----------------------

- Reward curve is plotted (moving average recommended).
- Safety curve is plotted (moving unsafe/failure/cost rate).
- Figures and tabular metrics are saved.

9. Reproducibility
------------------

- Random seeds fixed (env, numpy, torch, algorithm).
- Hyperparameters recorded.
- Shield and sensor definitions versioned.

10. Preflight Sanity Checks
---------------------------

- Sensor outputs verified on known states.
- Shield behavior checked on known unsafe actions.
- Short smoke run completes before long training.

