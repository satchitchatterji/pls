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

\[
\hat{y}_{i,j}=\sigma(z_{i,j})=\frac{1}{1+e^{-z_{i,j}}}
\]

Elementwise binary cross-entropy:

\[
\ell_{i,j}=-\left(y_{i,j}\log\hat{y}_{i,j}+(1-y_{i,j})\log(1-\hat{y}_{i,j})\right)
\]

Mean reduction used by default:

\[
\mathcal{L}=\frac{1}{4B}\sum_{i=1}^{B}\sum_{j=1}^{4}\ell_{i,j}
\]

Equivalent numerically stable logits form:

\[
\ell_{i,j}=\max(z_{i,j},0)-z_{i,j}y_{i,j}+\log\left(1+e^{-|z_{i,j}|}\right)
\]

Checkpoint output:

- `frozenlake_sensor_mlp.pt`

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

- `examples/images/frozenlake/`
