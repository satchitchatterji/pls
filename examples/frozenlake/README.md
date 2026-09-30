# FrozenLake Experiments in CleanPLS

This folder contains minimal, research-oriented FrozenLake experiments showing how to use probabilistic logic shielding with either oracle sensors or pretrained MLP sensors.

## Environment

We use Gymnasium **FrozenLake-v1** (4x4 map by default), where the agent starts at `S`, tries to reach `G`, and must avoid holes `H`.

- Official environment docs: <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>
- Environment animation:

![FrozenLake environment](https://gymnasium.farama.org/_images/frozen_lake.gif)

### State and Action

- State: discrete tile index in `{0, ..., 15}` for a 4x4 grid.
- Actions (fixed order used here):
  1. `left`
  2. `down`
  3. `right`
  4. `up`

The scripts use this action order consistently in sensor outputs and shield rules.

## Safety Signal

Safety in these examples is based on **hole transitions**:

- Unsafe transition: taking an action whose intended next tile is a hole.
- Episode failure proxy in logging: final reward `0.0` is counted as failure (did not reach goal).

For shielding, we use per-action safety indicators/probabilities:

- `left_to_hole`
- `down_to_hole`
- `right_to_hole`
- `up_to_hole`

Each value is interpreted as a probability in `[0,1]` that the corresponding action leads to a hole.

## Shield Definition (ProbLog)

The shield program marks actions unsafe when the corresponding sensor indicates hole risk, then derives safe actions by negation.

Conceptually:

- `unsafe_next(left) :- left_to_hole.`
- `unsafe_next(down) :- down_to_hole.`
- `unsafe_next(right) :- right_to_hole.`
- `unsafe_next(up) :- up_to_hole.`
- `safe_next(A) :- action(A), \+ unsafe_next(A).`

During action selection, the shield uses these rules to reduce probability mass on unsafe actions and renormalize the policy.

## Sensor Variants

### 1) Oracle sensor

Used in `frozenlake_oracle.py`.

- Deterministically computes hole-risk sensors from map geometry.
- Effectively perfect sensing (idealized upper bound for shield quality).

### 2) Pretrained MLP sensor

Used in `frozenlake_pretrain.py` + `frozenlake_pretrained.py`.

- Learns to predict action-to-hole risks from state features.
- Produces probabilistic outputs (after sigmoid) that can be passed directly to the shield.

## Pretraining Pipeline (MLP)

Implemented in `frozenlake_pretrain.py`.

### Data generation

For each state `s` in the 4x4 grid and each action `a` in `{left,down,right,up}`:

1. Compute intended next state `s'` with boundary clamping.
2. Label action as `1` if `s'` is a hole, else `0`.

Input features:

- one-hot state vector `x \in R^{16}`.

Targets:

- binary risk vector `y \in {0,1}^4` in action order `[left, down, right, up]`.

### MLP architecture

The default architecture is:

- `Linear(16,16) + ReLU`
- `Linear(16,16) + ReLU`
- `Linear(16,4)`

The 4 outputs are logits for the four action risk sensors.

### Loss function (BCEWithLogits)

Let batch size be `B`, logits `z_{i,j}`, labels `y_{i,j}` with `j in {1,...,4}`.

Sigmoid probability:

$$
\hat{y}_{i,j}=\sigma(z_{i,j})=\frac{1}{1+e^{-z_{i,j}}}
$$

Elementwise binary cross-entropy:

$$
\ell_{i,j}=-\left(y_{i,j}\log\hat{y}_{i,j}+(1-y_{i,j})\log(1-\hat{y}_{i,j})\right)
$$

Mean reduction used by default:

$$
\mathcal{L}=\frac{1}{4B}\sum_{i=1}^{B}\sum_{j=1}^{4}\ell_{i,j}
$$

Equivalent numerically stable logits form:

$$
\ell_{i,j}=\max(z_{i,j},0)-z_{i,j}y_{i,j}+\log\left(1+e^{-|z_{i,j}|}\right)
$$

Checkpoint output:

- `examples/frozenlake/frozenlake_sensor_mlp.pt`

## How Everything Ties Together

### Oracle path (no pretraining)

1. Build FrozenLake env.
2. Compute oracle action-risk sensors from state.
3. Feed sensors into ProbLog shield.
4. Train `PPO_shielded` (SPPO).
5. Save reward/failure curves.

Script:

- `frozenlake_oracle.py`

### Pretrained path

1. Run `frozenlake_pretrain.py` to train and save MLP sensor `.pt`.
2. Run `frozenlake_pretrained.py`.
3. Load checkpoint into sensor wrapper (`predict(obs, info=None)`).
4. Feed MLP sensor outputs into the same ProbLog shield.
5. Train `PPO_shielded` and save curves.

Scripts:

- `frozenlake_pretrain.py`
- `frozenlake_pretrained.py`

## Files in This Folder

- `frozenlake_oracle.py`: SPPO with oracle sensors, no registry/config.
- `frozenlake_mlp.py`: single-file SPPO with inline MLP pretraining.
- `frozenlake_pretrain.py`: standalone MLP pretraining + `.pt` export.
- `frozenlake_pretrained.py`: SPPO using loadable pretrained `.pt` sensor.
- `frozenlake_compare_oracle_vs_pretrained_mlp.ipynb`: PPO/A2C/SPPO/SA2C comparisons.
- `frozenlake_dqn_sdqn_variant_matrix.ipynb`: DQN/SDQN comparison matrix.

## Typical Run Order

```bash
python examples/frozenlake/frozenlake_oracle.py
python examples/frozenlake/frozenlake_pretrain.py
python examples/frozenlake/frozenlake_pretrained.py
```

Generated figures are saved under:

- `examples/frozenlake/images/`

## Optional JAX Sensor Backend

The file `frozenlake_jax.py` pretrains the same one-hot-state MLP using JAX
and saves a portable `.npz` checkpoint. It is a parallel sensor implementation
for backend experiments; it does not replace the Torch/SB3 shielded algorithms.
The resulting checkpoint can be loaded with
`pls.jax.FrozenLakeMLPSensorModel.from_checkpoint(...)`, whose `predict(obs,
info=None)` method follows the generic CleanPLS sensor interface.

For a fully JAX-native sensor-to-shield path, use
`FrozenLakeJaxSensorModel` together with `FrozenLakeJaxShield`. The sensor
returns JAX arrays, and the shield masks and renormalizes the policy on the
same JAX device. This avoids the CPU Torch conversion used only by the
compatibility wrapper for the existing SB3 shield.

Install the optional backend with:

```bash
pip install -e '.[jax]'
```

For an M4 Mac, install the Metal plugin as well:

```bash
pip install -e '.[jax-macos]'
```

This extra pins the tested Apple Silicon combination `jax==0.5.0`,
`jaxlib==0.5.0`, and `jax-metal==0.1.1`. The Metal plugin is experimental, and
using an unpinned newer JAX release can allow device discovery to succeed while
training compilation still fails.

Then run the sensor pretraining example from the repository root:

```bash
python examples/frozenlake/frozenlake_jax.py
```

To benchmark the identical MLP training workload in Torch and JAX on CPU and
available Apple GPU backends:

```bash
python examples/frozenlake/benchmark_torch_vs_jax.py
```

The benchmark uses Torch `mps` and JAX `metal` when available on Apple
Silicon. Missing or incompatible accelerators are reported as skipped, and
the results are saved as CSV plus a timing plot in `images/`.

## Device-Resident JAX Pipeline

`benchmark_jax_gpu_pipeline.py` extends the sensor-only comparison into a
fully JAX-native accelerator workload. During each timed update, the selected
JAX device handles:

- batched FrozenLake transitions,
- sensor inference,
- actor and critic forward passes,
- action sampling,
- shield masking and policy renormalization,
- generalized advantage estimation,
- PPO-style policy, value, entropy, and safety losses,
- Adam updates.

Only setup, final scalar conversion, and artifact writing remain on the host.
The benchmark uses deterministic 4x4 transitions equivalent to the map used by
the sensor labels, rather than stepping Gymnasium's Python environment inside
the timed loop.

Run it from the repository root:

```bash
python examples/frozenlake/benchmark_jax_gpu_pipeline.py
```

The current M4 reference run used 100 updates, 256 parallel environments, and
16 rollout steps. It measured approximately 1.80 ms/update on JAX CPU and
5.70 ms/update on JAX Metal. The CPU result is faster here because this tiny
workload does not provide enough parallel work to amortize Metal dispatch and
compilation overhead. Larger maps, batches, or networks are more appropriate
for assessing accelerator scaling.

Outputs:

- `images/frozenlake_jax_gpu_pipeline_comparison.csv`
- `images/frozenlake_jax_gpu_pipeline_comparison.png`
