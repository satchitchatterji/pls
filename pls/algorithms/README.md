# Shielded DRL Objectives (Core Math)

This note summarizes the optimization targets used in CleanPLS and how the probabilistic logic safety term is added.

## Notation

- State $s_t$, action $a_t$, reward $r_t$, next state $s_{t+1}$.
- Policy $\pi_\theta(a\mid s)$, value $V_\phi(s)$, Q-value $Q_\theta(s,a)$.
- Shield-derived policy safety:
  $$P_{\pi}(\mathrm{safe}\mid s) \in (0,1]$$
- Safety coefficient: $\alpha \ge 0$.
- Safety penalty:
  $$\mathcal{L}_{\mathrm{safe}}(s)= -\log\!\big(P_{\pi}(\mathrm{safe}\mid s)+\varepsilon\big)$$
  with small $\varepsilon>0$ for numerical stability.

A common template in this repo is:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{base}} + \alpha\,\mathcal{L}_{\mathrm{safe}}.
$$

---

## PPO (implemented)

Base PPO objective (minimization form):
$$
\mathcal{L}_{\mathrm{PPO}} = \mathcal{L}_{\mathrm{clip}} + c_v\,\mathcal{L}_V - c_e\,\mathcal{H}(\pi_\theta).
$$

Shielded PPO in CleanPLS:
$$
\mathcal{L}_{\mathrm{SPPO}} = \mathcal{L}_{\mathrm{PPO}} + \alpha\,\mathcal{L}_{\mathrm{safe}}.
$$

---

## DQN and PLTD (implemented)

TD target:
$$
y_t = r_t + \gamma\,\mathcal{X}_t.
$$

- Off-policy: $\mathcal{X}_t = \max_{a'} Q_{\bar\theta}(s_{t+1},a')$.
- On-policy approximation: $\mathcal{X}_t = Q_{\bar\theta}(s_{t+1},a_{t+1})$.

Base TD loss:
$$
\mathcal{L}_{\mathrm{TD}} = \mathbb{E}\big[(y_t-Q_\theta(s_t,a_t))^2\big]
$$
(or Huber in implementation).

PLTD style objective:
$$
\mathcal{L}_{\mathrm{SDQN}} = \mathcal{L}_{\mathrm{TD}} + \alpha\,\mathcal{L}_{\mathrm{safe}}
$$
which is equivalent to adding $-\alpha\log P_{\pi}(\mathrm{safe}\mid s)$ in reward-maximization form.

---

## Double DQN (implemented)

Double DQN target decouples action selection and evaluation:
$$
a^* = \arg\max_{a'}Q_\theta(s_{t+1},a'),\qquad
y_t = r_t + \gamma Q_{\bar\theta}(s_{t+1},a^*).
$$

Shielded form in this repo:
$$
\mathcal{L}_{\mathrm{SDoubleDQN}} = \mathcal{L}_{\mathrm{DoubleTD}} + \alpha\,\mathcal{L}_{\mathrm{safe}}.
$$

---

## Rainbow (implemented as Rainbow-lite)

Full Rainbow typically combines Double DQN + Dueling + Prioritized Replay + Multi-step + Distributional RL + Noisy Nets.

Current CleanPLS implementation is **Rainbow-lite**:
- built on the Double-DQN shielded target,
- exposes a Rainbow-compatible API surface,
- keeps the same safety inclusion term.

Objective:
$$
\mathcal{L}_{\mathrm{SRainbowLite}} = \mathcal{L}_{\mathrm{DoubleTD}} + \alpha\,\mathcal{L}_{\mathrm{safe}}.
$$

---

## SAC (implemented)

Standard SAC objective (actor-critic, entropy-regularized):
$$
J_Q(\theta_i)=\mathbb{E}\Big[\big(Q_{\theta_i}(s,a)-\big(r+\gamma(\min_j Q_{\bar\theta_j}(s',a')-\beta\log\pi(a'\mid s'))\big)\big)^2\Big]
$$
$$
J_\pi(\phi)=\mathbb{E}\big[\beta\log\pi_\phi(a\mid s)-Q_\theta(s,a)\big].
$$

Safety-augmented actor objective:
$$
J_{\pi}^{+}(\phi)=J_\pi(\phi)+\alpha\,\mathbb{E}[\mathcal{L}_{\mathrm{safe}}(s)].
$$

Implemented in this repo by adding the safety term directly to the actor loss in `train()`.
For Box actions, a differentiable categorical proxy is used:
$$
\tilde{\pi}(a\mid s)=\mathrm{softmax}(u_\phi(s)/\tau),
$$
where $u_\phi(s)$ is the actor output vector and $\tau$ is a temperature constant.
The shield then evaluates $P_{\tilde{\pi}}(\mathrm{safe}\mid s)$.

---

## TD3 (implemented)

Base TD3 critic objective:
$$
J_Q(\theta_i)=\mathbb{E}\big[(Q_{\theta_i}(s,a)-y)^2\big],
$$
$$
y=r+\gamma\min_{i=1,2}Q_{\bar\theta_i}(s',\pi_{\bar\phi}(s')+\epsilon),\quad \epsilon\sim\text{clip}(\mathcal{N}(0,\sigma),-c,c).
$$

Actor objective:
$$
J_\pi(\phi)= -\mathbb{E}[Q_{\theta_1}(s,\pi_\phi(s))].
$$

Safety-augmented actor objective:
$$
J_{\pi}^{+}(\phi)=J_\pi(\phi)+\alpha\,\mathbb{E}[\mathcal{L}_{\mathrm{safe}}(s)].
$$

Implemented in this repo by adding the safety term to delayed actor updates in `train()`, using the same softmax action-proxy relaxation for Box actions.

---

## DDPG (implemented)

Base DDPG actor objective:
$$
J_\pi(\phi)= -\mathbb{E}[Q_\theta(s,\pi_\phi(s))],
$$
with TD critic target as in deterministic actor-critic.

Safety-augmented actor objective:
$$
J_{\pi}^{+}(\phi)=J_\pi(\phi)+\alpha\,\mathbb{E}[\mathcal{L}_{\mathrm{safe}}(s)].
$$

Implemented in this repo by adding the safety term to actor updates in `train()`, using the same softmax action-proxy relaxation for Box actions.

---

## TRPO (implemented; requires `sb3-contrib`)

TRPO solves:
$$
\max_\theta\;\mathbb{E}_{s,a\sim\pi_{\theta_{\text{old}}}}\!\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}A^{\pi_{\theta_{\text{old}}}}(s,a)\right]
$$
subject to a KL trust-region constraint
$$
\mathbb{E}_s\big[D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}}(\cdot\mid s)\|\pi_\theta(\cdot\mid s))\big] \le \delta.
$$

Safety-augmented objective:
$$
\max_\theta\;\mathcal{J}_{\mathrm{TRPO}}(\theta)-\alpha\,\mathbb{E}[\mathcal{L}_{\mathrm{safe}}(s)]
$$
(or minimization equivalent with $+\alpha\mathcal{L}_{\mathrm{safe}}$).

Implemented in this repo directly in TRPO training by:
- optimizing the safety-augmented surrogate objective,
- enforcing the same KL trust-region constraint,
- and applying the line-search accept criterion on the augmented objective.

## Practical note

For all wrappers, the API is aligned with existing shielded classes:
- `alpha`
- `shield_params`
- `policy_safety_params`
- `config_folder`
- `get_sensor_value_ground_truth`

When a policy does not expose a discrete action distribution, the implementation uses a softmax relaxation over actor outputs when dimensions match `shield.num_actions`; otherwise the safety term is skipped.
