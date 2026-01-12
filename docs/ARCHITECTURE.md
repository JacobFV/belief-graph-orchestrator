# architecture

## 1. thesis

a real gui agent should not be modeled as a flat policy \( \pi(a_t \mid x_{\le t}) \) over screenshots. screenshots are not state. the environment is partially observed. the control interface is relative. task correctness is multi-scale. verification is mandatory. the true runtime state includes immutable evidence, revisable beliefs, active hypotheses, hidden control variables, and process-local workspaces.

therefore the brain is an **event-sourced, multi-scale, epistemic control system**.

let:
- \( \mathcal{E}_{\le t} \): immutable event store,
- \( \mathcal{G}^{(\ell)}_t \): belief graph slice at scale \(\ell\),
- \( \mathcal{R}_t \subset \mathcal{G} \): resident hot graph,
- \( \mathcal{Q}^{(k,\ell)}_t \subset \mathcal{R}_t \): query aperture for complex \(k\) at scale \(\ell\),
- \( \hat p_t \): pointer posterior,
- \( \mathcal{B}_t \): branch set,
- \( \Xi_t \): cross-scale contracts.

then

\[
S_t = (\mathcal{E}_{\le t}, \{\mathcal{G}^{(\ell)}_t\}_{\ell=0}^{5}, \mathcal{R}_t, \{\mathcal{Q}^{(k,\ell)}_t\}_{k,\ell}, \hat p_t, \mathcal{B}_t, \Xi_t)
\]

is the correct state object.


## process graph

```text
frames/task/acks
      │
      ▼
append-only event journal
      │
      ▼
perception -> evidence events
      │
      ▼
belief update -> graph revisions + branch support/contradiction
      │
      ▼
resident memory + query aperture assembly
      │
      ▼
scheduler emits (complex, scale) ticks
      │
      ├── pointer/action execution
      ├── verifier
      ├── task/subtask/interaction
      ├── recovery
      ├── historical reflection
      └── background maintenance
      │
      ▼
action compiler -> HID emission -> future evidence
```

## coarse-to-fine and fine-to-coarse

```text
L5 task / episode
  ↓ constrain
L4 subtask / route
  ↓ constrain
L3 interaction / target family
  ↓ constrain
L2 gesture chunk
  ↓ constrain
L1 servo control
  ↓ execute
L0 micro residuals

then bottom-up:
L0/L1 mismatch -> L2 gesture failure? -> L3 wrong target? -> L4 wrong route? -> L5 wrong plan?
```

## 2. scales

the runtime has six canonical scales:

\[
L_5 = \text{overall task / episode}
\]
\[
L_4 = \text{subtask / route / workflow chunk}
\]
\[
L_3 = \text{interaction / target selection}
\]
\[
L_2 = \text{gesture / action chunk}
\]
\[
L_1 = \text{servo / motor control}
\]
\[
L_0 = \text{microcontrol / residual timing}
\]

these are not merely “fast” and “slow.” they correspond to different latent interaction dynamics, graph radii, abstraction levels, and failure modes.

use scale coordinates

\[
\sigma = (\tau,\rho,\eta,\kappa,\chi)
\]

with:
- \(\tau\): temporal horizon,
- \(\rho\): graph/spatial radius,
- \(\eta\): semantic abstraction depth,
- \(\kappa\): control bandwidth,
- \(\chi\): branching width.

## 3. evidence vs belief

evidence is immutable. beliefs are revisable.

an event is

\[
e_i = (\mathrm{id}_i, \mathrm{type}_i, t_i^{\mathrm{cap}}, t_i^{\mathrm{arr}}, \mathrm{payload}_i, \mathrm{parents}_i, \Sigma_i)
\]

with separate capture and arrival times because latency matters.

belief nodes are versioned claims:

\[
v = (\mathrm{id}, \ell, \mathrm{kind}, \mathrm{version}, \mathrm{support}, \mathrm{contradict}, [t^-_v,t^+_v), c_v, \theta_v, \phi_v)
\]

where \(c_v\) is confidence, \(\theta_v\) is symbolic state, and \(\phi_v\) is a learned latent bundle.

## 4. graph families

the full graph is multi-resolution:

\[
\mathcal{G} = \bigcup_{\ell=0}^{5} \mathcal{G}^{(\ell)}
\]

node families by scale:
- \(L_0\): micro residuals, instantaneous pointer candidates, timing artifacts,
- \(L_1\): pointer posterior, local target envelope, control residual state,
- \(L_2\): gesture chunk, gesture phase, local expected feedback,
- \(L_3\): affordance, text span, candidate target, local container,
- \(L_4\): screen region, route hypothesis, subtask state,
- \(L_5\): goal state, historic anchor, analogy anchor, failure motif, episode summary.

edges include support, contradiction, containment, same-entity continuity, causality, prediction, resolution, summarization, refinement, and retrieval relevance.

## 5. existence / residency / accessibility / attention

a node can exist without being resident, be resident without being queryable by a given complex, be queryable without being packed into the actual workspace. these distinctions are real and should be implemented.

hotness score:

\[
h_t(v) = w_r r_t(v) + w_f f_t(v) + w_b b_t(v) + w_m m_t(v) + w_h h_t^{\mathrm{hist}}(v) + w_a a_t(v)
\]

controls promotion into the resident graph, but does not alter durable truth.

## 6. contracts

top-down contracts:

\[
C_5 = (\text{goal family}, \text{success criteria}, \text{risk posture})
\]
\[
C_4 = (\text{route hypothesis}, \text{region priors}, \text{expected transitions})
\]
\[
C_3 = (\text{candidate targets}, \text{interaction mode}, \text{local expectations})
\]
\[
C_2 = (\text{gesture phases}, \text{target distribution}, \text{timing/gain policy})
\]
\[
C_1 = (\text{velocity commands}, \text{pointer posterior}, \text{contact policy})
\]

upward messages report pointer error, motor feedback, gesture success, interaction outcome, route contradiction, and task progress.

## 7. pointer subsystem

the pointer is a hidden state:

\[
p_t = (x_t, y_t, \dot x_t, \dot y_t)
\]

with dynamics

\[
p_{t+1} = f_\theta(p_t, u_t, \Delta t_t) + \epsilon_t
\]

and visual measurements

\[
o_t \sim g_\psi(p_t, x_t) + \nu_t.
\]

the runtime maintains a posterior \(\hat p_t\) and a controller that drives the posterior toward a target distribution. it is therefore a state estimator plus controller, not just another classifier head.

## 8. action chunks and branches

action chunks are explicit objects carrying:
- action kind,
- target distribution,
- phase plan,
- velocity profile,
- timeout,
- expectation bundle,
- fallback policy.

each nontrivial action spawns branch families:

\[
\mathcal{B}(a_t) = \{b_{t,1},\dots,b_{t,K}\}
\]

with priors \(\pi_{t,k}\). the verifier updates branch posteriors from future evidence.

## 9. verifier

the verifier is multi-scale. it judges:
- motor/servo correctness,
- gesture correctness,
- interaction correctness,
- subtask progress,
- task progress.

this allows correct recovery selection: a pointer failure is not the same as a route failure.

## 10. complexes and scheduler

the runtime is a society of complexes:
- pointer,
- action execution,
- frontier visual,
- verifier,
- task,
- recovery,
- historical reflection,
- background.

each complex has a scale band, query budget, temporal stance, energy, and anchors. the scheduler allocates time and scale occupancy while respecting hazard and mandatory fast loops.

conceptually it solves

\[
\max_{b_{k,\ell}\ge 0}
\sum_{k,\ell}
 b_{k,\ell}(t)
 \big(U_{k,\ell}(t) - \lambda_C C_{k,\ell}(t) - \lambda_D D_{k,\ell}(t)\big)
\]

subject to compute constraints and hard control constraints.

## 11. reflection

history is not always bad. historical retrieval is permitted according to value-of-retrospection:

\[
\mathrm{VoR}_{k,\ell}(t) = \mathbb{E}[\Delta U \mid \text{retrieve history}] - \lambda_T \Delta T - \lambda_C \Delta C
\]

and suppressed when frontier hazard is too high.

## 12. implementation mapping

- `journal.py`: evidence substrate,
- `graph.py`: belief graph,
- `memory.py`: resident hotness,
- `perception.py`: evidence producers,
- `belief.py`: graph revision,
- `retrieval.py`: query apertures,
- `task.py`: subgoal / target selection,
- `compiler.py`: action chunks and branches,
- `pointer.py`: hidden-state control,
- `verifier.py`: multi-scale judgment,
- `recovery.py`: failure handling,
- `reflection.py`: historical/analogical retrieval,
- `scheduler.py`: complex-scale scheduling,
- `training/*`: replay, synthetic data, datasets, losses, trainer, eval.

this is the runtime skeleton that can be ported to swift later without losing the conceptual decomposition.
