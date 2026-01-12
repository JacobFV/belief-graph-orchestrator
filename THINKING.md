good. now we stop talking philosophy and turn it into an implementable on-device system.

i’m going to specify this as a concrete architecture for a single-device runtime with explicit processes, storage, data schemas, loop rates, model boundaries, retrieval, scheduling, pointer control, verification, and rollout phases. i’ll keep it pythonic / pseudocode-heavy so the structure is legible and portable.

the guiding constraints are:

* everything important must run on device
* the runtime cannot assume simulator privileges
* the runtime should work under partial observability
* the fast loops must stay cheap
* the large / slow models only see curated workspaces, never the full graph
* the system must support deterministic replay and postmortem training
* the graph itself can remain epistemically uncompressed; what changes is residency and accessibility

======================================================================

1. top-level runtime picture
   ======================================================================

think of the runtime as five interacting strata:

1. capture + eventization
2. perception + local tracking
3. belief graph + memory residency
4. multi-rate cognition + scheduling
5. action execution + pointer servo + verification

ascii block diagram:

```text
┌──────────────────────────────────────────────────────────────────┐
│                        ON-DEVICE AGENT                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [screen frames] [task text] [hid tx ack] [time]                 │
│        │             │            │         │                    │
│        └──────┬──────┴──────┬─────┴─────────┘                    │
│               ▼             ▼                                    │
│        ┌──────────────────────────┐                              │
│        │   event journal / bus    │  append-only                │
│        └────────────┬─────────────┘                              │
│                     ▼                                            │
│        ┌──────────────────────────┐                              │
│        │  perception subsystem    │                              │
│        │  - ocr                   │                              │
│        │  - region proposals      │                              │
│        │  - layout/container      │                              │
│        │  - pointer candidates    │                              │
│        │  - delta/motion          │                              │
│        └────────────┬─────────────┘                              │
│                     ▼                                            │
│        ┌──────────────────────────┐                              │
│        │  belief update engine    │                              │
│        │  typed time-native graph │                              │
│        └────────────┬─────────────┘                              │
│                     ▼                                            │
│        ┌──────────────────────────┐                              │
│        │ resident memory manager  │                              │
│        │ + historic anchors       │                              │
│        └──────┬───────────┬───────┘                              │
│               ▼           ▼                                      │
│   ┌────────────────┐  ┌──────────────────┐                       │
│   │ scheduler      │  │ query assembler  │                       │
│   └───┬─────┬──────┘  └────┬─────────────┘                       │
│       │     │              │                                     │
│       ▼     ▼              ▼                                     │
│  fast loops medium loops  slow loops                             │
│       │     │              │                                     │
│       ├─────┼───────┬──────┤                                     │
│       ▼     ▼       ▼      ▼                                     │
│  pointer  verifier task  reflection/recovery                     │
│  servo    loop     loop    loop                                  │
│       │               │                                           │
│       └──────┬────────┘                                           │
│              ▼                                                    │
│        ┌──────────────────────────┐                               │
│        │  action compiler         │                               │
│        │  + gesture chunker       │                               │
│        └────────────┬─────────────┘                               │
│                     ▼                                             │
│        ┌──────────────────────────┐                               │
│        │  hid velocity emitter    │                               │
│        └──────────────────────────┘                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

this is the whole machine.

======================================================================
2. process model
================

run the system as a small set of always-on actors / threads / async workers. don’t make it a single monolith. the control-critical pieces must stay isolated.

minimal runtime workers:

* `FrameIngestWorker`
* `PerceptionWorker`
* `BeliefWorker`
* `SchedulerWorker`
* `PointerServoWorker`
* `VerifierWorker`
* `TaskWorker`
* `RecoveryWorker`
* `BackgroundWorker`
* `ActionEmitterWorker`

do not put everything in one event loop.

recommended responsibility split:

```text
FrameIngestWorker:
    - receives screenshot/frame samples
    - stamps capture/arrival times
    - writes FrameEvent

PerceptionWorker:
    - subscribes to recent FrameEvents
    - runs OCR / region / layout / pointer candidate / delta
    - writes evidence events

BeliefWorker:
    - subscribes to evidence + action + verifier events
    - updates graph
    - updates active branches
    - updates resident set scores

SchedulerWorker:
    - computes which complexes tick now
    - assigns query budgets
    - may preempt / re-anchor complexes

PointerServoWorker:
    - high rate
    - tracks hidden pointer state
    - emits HID velocity commands during active action chunks

VerifierWorker:
    - scores expectation satisfaction
    - updates branch posteriors
    - emits success/failure/ambiguous judgments

TaskWorker:
    - slower
    - assembles query pool
    - chooses subgoal / target / action intent
    - requests action compilation

RecoveryWorker:
    - monitors failures, oscillation, stale fixation, divergence
    - injects recovery actions or strategy shifts

BackgroundWorker:
    - low rate
    - residency promotion/demotion
    - index maintenance
    - historic anchor scoring
    - dynamics adaptation
    - trace packaging
```

======================================================================
3. event model
==============

everything enters the system as an event. do not skip this. if a subsystem computed something important and it did not emit an event, you are weakening replay, learning, and debugging.

core schema:

```python
from dataclasses import dataclass, field
from typing import Any, Optional, Literal

EventType = Literal[
    "frame",
    "ocr",
    "regions",
    "layout",
    "pointer_candidates",
    "delta",
    "task_instruction",
    "action_issued",
    "hid_ack",
    "belief_revision",
    "branch_update",
    "verifier_judgment",
    "recovery_trigger",
    "teacher_structure",
    "metric"
]

@dataclass
class Event:
    id: int
    type: EventType
    t_capture_ns: int
    t_arrival_ns: int
    session_id: str
    episode_id: str
    parent_ids: list[int] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
```

do not try to store only one “state snapshot” every loop. store the stream.

minimal append-only journal API:

```python
class EventJournal:
    def append(self, event: Event) -> None: ...
    def get(self, event_id: int) -> Event: ...
    def tail(self, n: int, types: Optional[set[str]] = None) -> list[Event]: ...
    def range(self, t0_ns: int, t1_ns: int, types: Optional[set[str]] = None) -> list[Event]: ...
```

======================================================================
4. belief graph model
=====================

the belief graph is typed, temporal, and revisable.

node kinds:

* observed_entity
* tracked_entity
* text_span
* container
* affordance
* pointer_hypothesis
* transition_hypothesis
* expectation
* counterfactual_entity
* task_subgoal
* failure_pattern
* episode_summary
* historic_anchor
* analogy_anchor

edge kinds:

* supports
* contradicts
* contains
* same_entity_as
* caused_by
* predicted_by
* resolved_by
* near_spatially
* next_in_time
* retrieved_for
* relevant_to
* in_branch

schema:

```python
NodeKind = Literal[
    "observed_entity",
    "tracked_entity",
    "text_span",
    "container",
    "affordance",
    "pointer_hypothesis",
    "transition_hypothesis",
    "expectation",
    "counterfactual_entity",
    "task_subgoal",
    "failure_pattern",
    "episode_summary",
    "historic_anchor",
    "analogy_anchor"
]

EdgeKind = Literal[
    "supports",
    "contradicts",
    "contains",
    "same_entity_as",
    "caused_by",
    "predicted_by",
    "resolved_by",
    "near_spatially",
    "next_in_time",
    "retrieved_for",
    "relevant_to",
    "in_branch"
]

@dataclass
class BeliefNode:
    id: int
    version: int
    kind: NodeKind
    status: Literal["active", "dormant", "superseded", "invalidated"]
    support_event_ids: list[int]
    support_node_ids: list[int]
    contradiction_event_ids: list[int]
    t_start_ns: int
    t_end_ns: Optional[int]
    confidence: float
    state: dict[str, Any]
    z_obj: Optional[list[float]] = None
    z_dyn: Optional[list[float]] = None
    z_belief: Optional[list[float]] = None
    z_value: Optional[list[float]] = None

@dataclass
class BeliefEdge:
    src: int
    dst: int
    kind: EdgeKind
    confidence: float
    t_start_ns: int
    t_end_ns: Optional[int]
```

graph API:

```python
class BeliefGraph:
    def add_node(self, node: BeliefNode) -> None: ...
    def add_edge(self, edge: BeliefEdge) -> None: ...
    def neighbors(self, node_id: int, kinds: Optional[set[EdgeKind]] = None) -> list[int]: ...
    def node(self, node_id: int) -> BeliefNode: ...
    def active_nodes(self) -> list[int]: ...
    def nodes_by_kind(self, kind: NodeKind) -> list[int]: ...
```

======================================================================
5. storage tiers vs query tiers
===============================

this distinction needs a hard implementation.

four layers:

* durable graph: all nodes/edges/events
* resident graph: currently memory-loaded subset
* per-complex query pool: currently accessible subset
* workspace registers: tiny active payload for an actual model call

ascii diagram:

```text
durable truth
┌────────────────────────────────────────────────────┐
│ all events + all graph nodes + all branches        │
└────────────────────────────────────────────────────┘
                    │ promotion / demotion
                    ▼
resident graph
┌────────────────────────────────────────────────────┐
│ hot recent frontier                                │
│ active task neighborhood                           │
│ unresolved branches                                │
│ pointer-related nodes                              │
│ useful historic anchors                            │
└────────────────────────────────────────────────────┘
                    │ per-complex retrieval
                    ▼
query pool for complex k
┌────────────────────────────────────────────────────┐
│ frontier subset                                    │
│ causal ancestors                                   │
│ semantic matches                                   │
│ analogical anchors                                 │
│ small branch slice                                 │
└────────────────────────────────────────────────────┘
                    │ packing / summarization
                    ▼
workspace
┌────────────────────────────────────────────────────┐
│ tokens / tensors fed to current model              │
└────────────────────────────────────────────────────┘
```

resident manager:

```python
@dataclass
class ResidencyRecord:
    node_id: int
    hot_score: float
    last_access_ns: int
    access_count_recent: int
    frontier_score: float
    branch_score: float
    motor_score: float
    historic_score: float
    anomaly_score: float
```

promotion score:

```python
def compute_hot_score(r: ResidencyRecord) -> float:
    return (
        1.2 * r.frontier_score +
        1.0 * r.branch_score +
        1.4 * r.motor_score +
        0.8 * r.historic_score +
        1.1 * r.anomaly_score +
        0.5 * min(r.access_count_recent, 10)
    )
```

then promote if score > threshold or part of a pinned subgraph.

======================================================================
6. on-device model stack
========================

do not try to cram a single huge general model into every loop.

use 5 model classes:

1. cheap perception trunk
2. cheap pointer observation model
3. cheap node scoring / retrieval model
4. medium semantic deliberation model
5. tiny residual control / verifier heads

concretely:

* model A: patch/image encoder for regions and layout hints
* model B: OCR/text detector/recognizer
* model C: pointer-candidate detector
* model D: task-conditioned node scorer / retrieval scorer
* model E: semantic deliberator over curated workspace
* model F: verifier head
* model G: control residual head

the exact mobile deployment format can vary, but architecturally:

```text
screen frame
  ├── model A -> patch / region features
  ├── model B -> text spans
  ├── model C -> pointer candidates
  └── light delta logic -> motion/change stats

belief nodes + task state + complex state
  ├── model D -> candidate node scores / retrieval scores
  └── model E -> subgoal / target / action intent / expectations

pointer posterior + target distribution
  └── model G -> residual control term

expectations + post-action evidence
  └── model F -> success/failure/ambiguous + branch likelihoods
```

important: the semantic deliberator should never look at the whole screenshot raw unless needed. it should mostly consume curated node/workspace summaries.

======================================================================
7. cognitive complexes
======================

make them concrete.

```python
ComplexKind = Literal[
    "pointer_servo",
    "action_execution",
    "frontier_visual",
    "verifier",
    "task_pursuit",
    "recovery",
    "background",
    "historical_reflection"
]

@dataclass
class QueryBudget:
    frontier_k: int
    spatial_k: int
    causal_k: int
    branch_k: int
    semantic_k: int
    historical_k: int
    analogical_k: int

@dataclass
class TemporalStance:
    frontier_pull: float
    history_pull: float
    counterfactual_pull: float
    novelty_pull: float

@dataclass
class ComplexState:
    id: str
    kind: ComplexKind
    tick_hz: float
    energy: float
    lock_strength: float
    anchor_node_ids: list[int]
    stance: TemporalStance
    budget: QueryBudget
    latent: dict[str, Any]
```

default complexes:

```python
POINTER_SERVO = ComplexState(
    id="pointer",
    kind="pointer_servo",
    tick_hz=60.0,
    energy=1.0,
    lock_strength=0.9,
    anchor_node_ids=[],
    stance=TemporalStance(1.0, 0.0, 0.05, 0.0),
    budget=QueryBudget(8, 8, 2, 1, 2, 0, 0),
    latent={}
)

TASK_PURSUIT = ComplexState(
    id="task",
    kind="task_pursuit",
    tick_hz=4.0,
    energy=1.0,
    lock_strength=0.6,
    anchor_node_ids=[],
    stance=TemporalStance(0.7, 0.35, 0.25, 0.1),
    budget=QueryBudget(16, 16, 8, 8, 16, 6, 4),
    latent={}
)

HISTORICAL_REFLECTION = ComplexState(
    id="reflect",
    kind="historical_reflection",
    tick_hz=1.0,
    energy=0.5,
    lock_strength=0.3,
    anchor_node_ids=[],
    stance=TemporalStance(0.2, 0.9, 0.5, 0.1),
    budget=QueryBudget(4, 4, 8, 4, 10, 16, 12),
    latent={}
)
```

======================================================================
8. scheduler
============

this needs to be explicit, not vibes.

state inputs:

* active action chunk?
* pointer uncertainty
* branch entropy
* recent verifier failures
* current complex energy
* task urgency
* current phase deadlines

scheduler output:

* which complexes tick
* their budgets
* any forced re-anchoring
* whether historical reflection is permitted now

pseudocode:

```python
class Scheduler:
    def step(self, now_ns: int, state: "RuntimeState") -> list[str]:
        ready = []
        for cx in state.complexes.values():
            if self._due(cx, now_ns):
                ready.append(cx.id)

        # hard-priority loops
        hard = []
        if "pointer" in ready:
            hard.append("pointer")
        if "verifier" in ready:
            hard.append("verifier")
        if state.active_action_chunk is not None and "action_exec" in ready:
            hard.append("action_exec")

        soft = [cid for cid in ready if cid not in hard]

        # suppress or allow reflective loops
        frontier_hazard = self.compute_frontier_hazard(state)
        vor = self.compute_value_of_retrospection(state)

        runnable = list(hard)

        for cid in soft:
            if cid == "historical_reflection":
                if vor > 0.0 and frontier_hazard < 0.6:
                    runnable.append(cid)
            else:
                runnable.append(cid)

        # optional re-anchoring
        self.reanchor_if_stale(state, frontier_hazard)

        return runnable

    def compute_frontier_hazard(self, state: "RuntimeState") -> float:
        return (
            0.45 * state.pointer_uncertainty +
            0.20 * state.branch_entropy +
            0.20 * state.fragile_action_phase +
            0.15 * state.pending_timeout_pressure
        )

    def compute_value_of_retrospection(self, state: "RuntimeState") -> float:
        return (
            0.35 * state.failure_density +
            0.25 * state.branch_entropy +
            0.20 * state.ambiguity_score +
            0.20 * state.analogy_match_score
            - 0.40 * state.fragile_action_phase
        )
```

this is simple enough to ship and later replace parts with learned policies.

======================================================================
9. query aperture assembly
==========================

do not just run nearest neighbors over all resident nodes. the retrieval should respect source families and budgets.

candidate families:

* frontier nodes
* target-adjacent spatial nodes
* recent causal ancestors
* unresolved branch nodes
* semantic matches to task text
* historical anchors
* analogy anchors
* anomaly-linked nodes

pseudocode:

```python
def assemble_query_pool(
    graph: BeliefGraph,
    resident_ids: set[int],
    complex_state: ComplexState,
    runtime: "RuntimeState"
) -> list[int]:
    frontier = score_frontier_nodes(graph, resident_ids, runtime, complex_state)
    spatial = score_spatial_nodes(graph, resident_ids, runtime, complex_state)
    causal = score_causal_nodes(graph, resident_ids, runtime, complex_state)
    branch = score_branch_nodes(graph, resident_ids, runtime, complex_state)
    semantic = score_semantic_nodes(graph, resident_ids, runtime, complex_state)
    historical = score_historical_nodes(graph, resident_ids, runtime, complex_state)
    analogical = score_analogical_nodes(graph, resident_ids, runtime, complex_state)

    selected = []
    selected += topk(frontier, complex_state.budget.frontier_k)
    selected += topk(spatial, complex_state.budget.spatial_k)
    selected += topk(causal, complex_state.budget.causal_k)
    selected += topk(branch, complex_state.budget.branch_k)
    selected += topk(semantic, complex_state.budget.semantic_k)
    selected += topk(historical, complex_state.budget.historical_k)
    selected += topk(analogical, complex_state.budget.analogical_k)

    return dedupe_preserve_order(selected)
```

scoring example:

```python
def score_semantic_nodes(graph, resident_ids, runtime, cx):
    task_vec = runtime.task_embedding
    scores = []
    for nid in resident_ids:
        node = graph.node(nid)
        if node.z_obj is None:
            continue
        s = dot(project(node.z_obj, "task"), task_vec)
        s += 0.2 * node.confidence
        s += 0.15 * node.state.get("actionable_prob", 0.0)
        scores.append((nid, s))
    return scores
```

======================================================================
10. perception concrete plan
============================

the perception layer should be as simple as possible while still producing structured evidence.

output structs:

```python
@dataclass
class OCRSpan:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float

@dataclass
class RegionProposal:
    bbox: tuple[float, float, float, float]
    score: float
    label_hint: Optional[str]
    patch_embedding: list[float]

@dataclass
class LayoutHint:
    kind: Literal["nav_bar", "tab_bar", "sheet", "modal", "list", "keyboard", "toolbar", "unknown"]
    bbox: tuple[float, float, float, float]
    confidence: float

@dataclass
class PointerCandidate:
    x: float
    y: float
    confidence: float
    visual_signature: list[float]

@dataclass
class DeltaFeatures:
    changed_regions: list[tuple[float, float, float, float]]
    global_change_score: float
    dominant_motion: tuple[float, float]
```

perception worker:

```python
class PerceptionWorker:
    def process_frame(self, frame_event: Event, prev_frames: list[Event]) -> list[Event]:
        img = frame_event.payload["image"]

        ocr_spans = self.ocr_model.run(img)
        proposals = self.region_model.run(img)
        layout = self.layout_model.run(img, proposals, ocr_spans)
        ptrs = self.pointer_model.run(img)
        delta = self.delta_model.run(img, [f.payload["image"] for f in prev_frames])

        return [
            mk_event("ocr", frame_event, {"spans": ocr_spans}),
            mk_event("regions", frame_event, {"proposals": proposals}),
            mk_event("layout", frame_event, {"layout_hints": layout}),
            mk_event("pointer_candidates", frame_event, {"pointer_candidates": ptrs}),
            mk_event("delta", frame_event, {"delta": delta}),
        ]
```

keep these models separate or separable even if they share a trunk.

======================================================================
11. belief update concrete plan
===============================

belief update pipeline:

1. ingest latest evidence
2. associate with existing tracked entities
3. create new observed/tracked nodes
4. update affordance posteriors
5. update containers and text attachments
6. update live branches / expectations
7. emit belief revision events
8. refresh resident hot scores

association logic:

```python
def associate_proposal_to_entities(
    proposal: RegionProposal,
    graph: BeliefGraph,
    candidate_entity_ids: list[int]
) -> Optional[int]:
    best = None
    best_score = -1e9
    for nid in candidate_entity_ids:
        n = graph.node(nid)
        s = 0.0
        s += 0.45 * cosine(n.z_obj, proposal.patch_embedding) if n.z_obj else 0.0
        s += 0.20 * iou(n.state.get("bbox"), proposal.bbox)
        s += 0.20 * role_compatibility(n.state.get("role_probs", {}), proposal.label_hint)
        s += 0.15 * temporal_continuity(n)
        if s > best_score:
            best_score = s
            best = nid
    return best if best_score > 0.55 else None
```

belief update step:

```python
class BeliefWorker:
    def step(self, new_events: list[Event], state: "RuntimeState") -> None:
        evidence = bucket_events(new_events)

        # 1. update entities from proposals
        for prop in evidence.get("regions", []):
            nid = associate_proposal_to_entities(prop, state.graph, state.recent_entity_candidates)
            if nid is None:
                self.create_new_entity(prop, state)
            else:
                self.revise_entity(nid, prop, state)

        # 2. attach OCR
        for span in evidence.get("ocr", []):
            self.attach_text_span(span, state)

        # 3. update layouts / containers
        for lh in evidence.get("layout", []):
            self.update_layout_container(lh, state)

        # 4. pointer hypotheses
        for pc in evidence.get("pointer_candidates", []):
            self.update_pointer_hypothesis(pc, state)

        # 5. branch support / contradiction
        self.update_live_branches(evidence, state)

        # 6. resident hotness refresh
        state.refresh_hot_scores()
```

======================================================================
12. branch / expectation system
===============================

every nontrivial action should generate branches.

example:

```python
@dataclass
class Expectation:
    kind: Literal[
        "target_state_change",
        "overlay_appears",
        "keyboard_appears",
        "scroll_displacement",
        "navigation_change",
        "text_change",
        "none_visible"
    ]
    target_node_ids: list[int]
    params: dict[str, Any]
    confidence: float

@dataclass
class Branch:
    id: int
    root_action_event_id: int
    prior: float
    posterior: float
    status: Literal["live", "supported", "rejected", "stale"]
    expectation_ids: list[int]
    node_ids: list[int]
```

action intent to branch bundle:

```python
def make_branches_for_tap(target_node_id: int, graph: BeliefGraph) -> list[Branch]:
    node = graph.node(target_node_id)

    role_probs = node.state.get("role_probs", {})
    branches = []

    if role_probs.get("ActionableDiscrete", 0.0) > 0.2:
        branches.append(branch_overlay_or_state_change(target_node_id, prior=0.45))
        branches.append(branch_navigation_change(target_node_id, prior=0.25))
        branches.append(branch_no_visible_effect(target_node_id, prior=0.15))
        branches.append(branch_keyboard_appears(target_node_id, prior=0.15))

    normalize_branch_priors(branches)
    return branches
```

verification should update branch posteriors, not just emit pass/fail.

======================================================================
13. semantic deliberation concrete plan
=======================================

the task worker should not run on every frame. it should run when:

* current action chunk completed or failed
* no active action chunk
* branch entropy too high
* uncertainty high enough to require reselection
* periodic refresh tick

workspace packing:

```python
@dataclass
class WorkspaceNode:
    node_id: int
    kind: str
    bbox: Optional[tuple[float, float, float, float]]
    text: Optional[str]
    confidence: float
    role_probs: dict[str, float]
    z_obj: Optional[list[float]]
    z_dyn: Optional[list[float]]
    z_belief: Optional[list[float]]

@dataclass
class Workspace:
    task_text: str
    task_embedding: list[float]
    pointer_summary: dict[str, Any]
    branch_summary: dict[str, Any]
    recent_action_summary: list[dict]
    nodes: list[WorkspaceNode]
```

deliberator output:

```python
@dataclass
class SubgoalDecision:
    description: str
    confidence: float

@dataclass
class TargetSelection:
    node_ids: list[int]
    target_distribution: dict[str, Any]
    confidence: float

@dataclass
class ActionIntent:
    kind: Literal["tap", "scroll", "drag", "type", "wait", "back", "dismiss", "explore"]
    params: dict[str, Any]
    confidence: float
```

deliberation pseudocode:

```python
class TaskWorker:
    def step(self, state: "RuntimeState") -> None:
        q = assemble_query_pool(state.graph, state.resident_ids, state.complexes["task"], state)
        ws = self.pack_workspace(q, state)
        out = self.deliberator.run(ws)

        state.current_subgoal = out.subgoal
        state.pending_target = out.target
        state.pending_intent = out.intent

        compiled = self.action_compiler.compile(out.intent, out.target, state)
        state.active_action_chunk = compiled
```

======================================================================
14. action compiler concrete plan
=================================

action chunks are explicit.

```python
@dataclass
class ActionPhase:
    name: str
    params: dict[str, Any]

@dataclass
class ActionChunk:
    id: str
    kind: Literal["tap", "scroll", "drag", "type", "wait", "back", "dismiss", "explore"]
    target_distribution: dict[str, Any]
    phases: list[ActionPhase]
    velocity_profile: dict[str, Any]
    timeout_ms: int
    expectation_ids: list[int]
    fallback_policy: dict[str, Any]
```

compile tap:

```python
def compile_tap(target: TargetSelection, state: "RuntimeState") -> ActionChunk:
    return ActionChunk(
        id=new_id("tap"),
        kind="tap",
        target_distribution=target.target_distribution,
        phases=[
            ActionPhase("approach", {"mode": "velocity_control"}),
            ActionPhase("slow_approach", {"radius_px": 24}),
            ActionPhase("contact", {"button": "primary"}),
            ActionPhase("dwell", {"ms": 60}),
            ActionPhase("release", {}),
            ActionPhase("verify", {"window_ms": 500}),
        ],
        velocity_profile={
            "far_gain": 1.0,
            "near_gain": 0.35,
            "max_vel": 1.0,
            "deadband_px": 3.0,
        },
        timeout_ms=1200,
        expectation_ids=create_expectations_for_tap(target, state),
        fallback_policy={"on_fail": "retarget_or_recover"}
    )
```

compile scroll:

```python
def compile_scroll(target: TargetSelection, direction: str, amount: float, state: "RuntimeState") -> ActionChunk:
    return ActionChunk(
        id=new_id("scroll"),
        kind="scroll",
        target_distribution=target.target_distribution,
        phases=[
            ActionPhase("approach", {"mode": "velocity_control"}),
            ActionPhase("contact", {"button": "primary"}),
            ActionPhase("drag", {"direction": direction, "amount": amount}),
            ActionPhase("release", {}),
            ActionPhase("verify_displacement", {"window_ms": 400}),
        ],
        velocity_profile={
            "drag_vel": 0.75,
            "release_damping": 0.2,
        },
        timeout_ms=1800,
        expectation_ids=create_expectations_for_scroll(target, direction, amount, state),
        fallback_policy={"on_fail": "rescan_scrollable_container"}
    )
```

======================================================================
15. pointer servo concrete plan
===============================

this is the high-rate hidden-state controller.

state:

```python
@dataclass
class PointerPosterior:
    x_hat: float
    y_hat: float
    vx_hat: float
    vy_hat: float
    cov: list[list[float]]
    visible_conf: float
    last_obs_event_id: Optional[int]
    dynamics: dict[str, float]
```

substeps every servo tick:

1. predict state using prior + issued velocity + dt
2. incorporate latest pointer observation candidate if any
3. if active action chunk, compute target error
4. compute velocity command
5. emit HID command
6. log action_issued event

predictor:

```python
def predict_pointer(pp: PointerPosterior, last_cmd: dict[str, float], dt_s: float) -> PointerPosterior:
    gain_x = pp.dynamics.get("gain_x", 1.0)
    gain_y = pp.dynamics.get("gain_y", 1.0)

    x = pp.x_hat + gain_x * last_cmd["vx"] * dt_s
    y = pp.y_hat + gain_y * last_cmd["vy"] * dt_s
    vx = last_cmd["vx"]
    vy = last_cmd["vy"]

    cov = inflate_cov(pp.cov, q=0.8)
    return PointerPosterior(x, y, vx, vy, cov, pp.visible_conf * 0.95, pp.last_obs_event_id, pp.dynamics)
```

corrector:

```python
def correct_pointer(pp: PointerPosterior, cand: Optional[PointerCandidate]) -> PointerPosterior:
    if cand is None or cand.confidence < 0.3:
        return pp

    alpha = min(max(cand.confidence, 0.1), 0.9)
    x = alpha * cand.x + (1 - alpha) * pp.x_hat
    y = alpha * cand.y + (1 - alpha) * pp.y_hat

    cov = shrink_cov(pp.cov, factor=(0.4 + 0.5 * cand.confidence))
    return PointerPosterior(x, y, pp.vx_hat, pp.vy_hat, cov, cand.confidence, None, pp.dynamics)
```

controller:

```python
def compute_velocity(pp: PointerPosterior, target_xy: tuple[float, float], profile: dict[str, Any]) -> tuple[float, float]:
    tx, ty = target_xy
    ex = tx - pp.x_hat
    ey = ty - pp.y_hat
    d = (ex * ex + ey * ey) ** 0.5

    if d < profile["deadband_px"]:
        return 0.0, 0.0

    gain = profile["far_gain"] if d > 40 else profile["near_gain"]

    vx = clip(gain * ex, -profile["max_vel"], profile["max_vel"])
    vy = clip(gain * ey, -profile["max_vel"], profile["max_vel"])

    # optional learned residual
    rvx, rvy = learned_residual(pp, target_xy)
    vx = clip(vx + rvx, -profile["max_vel"], profile["max_vel"])
    vy = clip(vy + rvy, -profile["max_vel"], profile["max_vel"])

    return vx, vy
```

servo loop:

```python
class PointerServoWorker:
    def tick(self, state: "RuntimeState", dt_s: float) -> None:
        pp = predict_pointer(state.pointer, state.last_hid_cmd, dt_s)

        cand = state.latest_pointer_candidate
        pp = correct_pointer(pp, cand)

        state.pointer = pp

        chunk = state.active_action_chunk
        if chunk is None:
            return

        phase = current_phase(chunk, state.now_ns)
        target_xy = sample_target_xy(chunk.target_distribution, pp)
        vx, vy = compute_velocity(pp, target_xy, chunk.velocity_profile)

        cmd = phase_to_hid_command(phase, vx, vy)
        state.last_hid_cmd = cmd
        emit_hid(cmd)
        log_action_issued(cmd, state)
```

======================================================================
16. verifier concrete plan
==========================

the verifier should monitor the latest action chunk’s expectations.

inputs:

* expectation bundle
* latest evidence events
* current graph state
* current branch priors

outputs:

* verdict
* branch posterior update
* failure trigger if needed

pseudocode:

```python
@dataclass
class VerifierVerdict:
    label: Literal["success", "partial", "failure", "ambiguous", "delayed"]
    score: float
    branch_posteriors: dict[int, float]
    notes: dict[str, Any]

class VerifierWorker:
    def tick(self, state: "RuntimeState") -> None:
        chunk = state.active_action_chunk
        if chunk is None:
            return

        exps = [state.graph.node(eid) for eid in chunk.expectation_ids]
        recent = state.event_journal.tail(32)

        verdict = self.run_verifier(exps, recent, state)

        log_verifier_judgment(verdict, state)

        if verdict.label in {"success", "partial"}:
            maybe_close_chunk(chunk, verdict, state)
        elif verdict.label == "failure":
            trigger_recovery("verification_failure", state)
```

a simple verifier can start rule-heavy. later it can use a model.

example expectation scoring:

```python
def score_expectation(exp: BeliefNode, recent_events: list[Event], state: "RuntimeState") -> float:
    k = exp.state["kind"]
    if k == "overlay_appears":
        return score_overlay_appears(exp, recent_events, state)
    if k == "keyboard_appears":
        return score_keyboard_appears(exp, recent_events, state)
    if k == "scroll_displacement":
        return score_scroll_displacement(exp, recent_events, state)
    if k == "target_state_change":
        return score_target_state_change(exp, recent_events, state)
    if k == "navigation_change":
        return score_navigation_change(exp, recent_events, state)
    return 0.0
```

======================================================================
17. historical reflection and value-of-retrospection
====================================================

this is not fluff. make it concrete.

the historical reflection complex runs slowly and only when justified. it retrieves old anchors or analogical episodes to help with diagnosis, disambiguation, or planning.

historic anchor schema:

```python
@dataclass
class HistoricAnchorState:
    summary_text: str
    utility_score: float
    analogy_signature: list[float]
    retrieval_count: int
    last_helped_ns: int
    linked_failure_patterns: list[int]
```

value-of-retrospection:

```python
def compute_vor(state: "RuntimeState") -> float:
    return (
        0.30 * state.failure_density +
        0.25 * state.branch_entropy +
        0.20 * state.ambiguity_score +
        0.20 * state.analogy_match_score -
        0.30 * state.fragile_action_phase -
        0.25 * state.pointer_uncertainty
    )
```

reflection step:

```python
class HistoricalReflectionWorker:
    def tick(self, state: "RuntimeState") -> None:
        if compute_vor(state) <= 0.0:
            return

        q = assemble_query_pool(state.graph, state.resident_ids, state.complexes["historical_reflection"], state)
        anchors = [nid for nid in q if state.graph.node(nid).kind in {"historic_anchor", "analogy_anchor"}]
        if not anchors:
            return

        insight = self.reflector.run(pack_reflection_workspace(anchors, state))
        apply_reflection_insight(insight, state)
```

the reflection output should be concrete, e.g. “expand target set to include nearby sibling,” “previous similar failure was caused by hidden overlay,” “prefer dismiss before retry,” not vague latent vibes.

======================================================================
18. recovery concrete plan
==========================

recovery triggers:

* repeated verification failure
* large pointer divergence
* oscillatory target switching
* stale unresolved branch
* timeout exceeded
* no visible effect after multiple retries
* probable hidden overlay
* scroll container mismatch

recovery policy skeleton:

```python
class RecoveryWorker:
    def tick(self, state: "RuntimeState") -> None:
        triggers = collect_recovery_triggers(state)
        if not triggers:
            return

        plan = self.choose_recovery_plan(triggers, state)
        enact_recovery_plan(plan, state)
```

plans:

* `reacquire_pointer`
* `slow_retarget_same_node`
* `expand_to_neighbor_candidates`
* `dismiss_overlay_then_retry`
* `back_then_rescan`
* `pause_and_wait`
* `scroll_to_reveal_more_context`
* `invoke_historical_reflection`

recovery choice can begin rule-based:

```python
def choose_recovery_plan(triggers, state):
    if "pointer_divergence" in triggers:
        return {"kind": "reacquire_pointer"}
    if "hidden_overlay_likely" in triggers:
        return {"kind": "dismiss_overlay_then_retry"}
    if "oscillation" in triggers:
        return {"kind": "slow_retarget_same_node"}
    if "ambiguous_branch" in triggers:
        return {"kind": "invoke_historical_reflection"}
    return {"kind": "pause_and_wait"}
```

======================================================================
19. training / replay hooks
===========================

every subsystem should emit data useful for later training.

what to log:

* raw frame references
* OCR outputs
* region proposals
* layout hints
* pointer candidates
* pointer posterior trajectory
* HID commands emitted
* action chunks
* branch priors/posteriors
* verifier verdicts
* recovery triggers/plans
* query pools presented to each complex
* semantic deliberator inputs/outputs

trace schema:

```python
@dataclass
class TrainingTrace:
    session_id: str
    event_ids: list[int]
    graph_node_ids: list[int]
    action_chunk_ids: list[str]
    verifier_judgments: list[dict]
    outcome_summary: dict[str, Any]
```

this lets you train:

* pseudo-structure inference
* pointer observation matching
* target ranking
* branch prediction
* verifier heads
* scheduler policies
* recovery policies
* value-of-retrospection

======================================================================
20. concrete on-device constraints and design rules
===================================================

since this should run entirely on device, enforce these rules:

rule 1:
fast loops only use tiny models or explicit math.

rule 2:
semantic deliberation frequency is low and input is tightly capped.

rule 3:
most frames do not trigger full graph-wide updates.

rule 4:
all retrieval is top-k and source-budgeted.

rule 5:
embeddings are cached on nodes; only dynamic parts are refreshed locally.

rule 6:
resident graph size is bounded independently from durable graph size.

rule 7:
every model call has a budgeted max input size.

rule 8:
the system should degrade gracefully:

* if OCR fails, still use layout and affordance priors
* if pointer candidates fail, rely more on prediction
* if semantic deliberation is skipped on a tick, fast loops continue

rule 9:
prefer recurrent / event-driven updates over recomputing everything from scratch.

rule 10:
support deterministic replay from on-device logs.

======================================================================
21. concrete runtime state object
=================================

```python
@dataclass
class RuntimeState:
    event_journal: EventJournal
    graph: BeliefGraph

    resident_ids: set[int]
    residency: dict[int, ResidencyRecord]

    complexes: dict[str, ComplexState]

    pointer: PointerPosterior
    latest_pointer_candidate: Optional[PointerCandidate]
    last_hid_cmd: dict[str, float]

    current_task_text: str
    task_embedding: list[float]
    current_subgoal: Optional[SubgoalDecision]

    active_action_chunk: Optional[ActionChunk]
    pending_target: Optional[TargetSelection]
    pending_intent: Optional[ActionIntent]

    live_branch_ids: list[int]

    pointer_uncertainty: float
    branch_entropy: float
    fragile_action_phase: float
    pending_timeout_pressure: float
    failure_density: float
    ambiguity_score: float
    analogy_match_score: float

    now_ns: int
```

======================================================================
22. full tick cycle
===================

this is the concrete runtime heartbeat.

```python
def main_runtime_tick(state: RuntimeState):
    # 1. ingest any new frame/task/hid ack events
    new_events = poll_inputs_and_append_events(state)

    # 2. perception for new frames
    perception_events = run_perception_if_needed(new_events, state)
    append_all(state.event_journal, perception_events)

    # 3. belief updates
    run_belief_update(new_events + perception_events, state)

    # 4. scheduler chooses runnable complexes
    runnable = scheduler.step(state.now_ns, state)

    # 5. run complexes in priority order
    for cid in runnable:
        run_complex(cid, state)

    # 6. background metrics / trace packaging
    maybe_emit_metrics(state)
```

priority order:

```python
PRIORITY = [
    "pointer_servo",
    "action_execution",
    "verifier",
    "frontier_visual",
    "recovery",
    "task_pursuit",
    "historical_reflection",
    "background",
]
```

======================================================================
23. diagrams for the two hardest loops
======================================

pointer loop:

```text
       last pointer posterior
               │
               ▼
        ┌──────────────┐
        │ predictor    │  uses last cmd + dt + dynamics
        └──────┬───────┘
               ▼
      predicted pointer state
               │
               ├───────────────┐
               │               │
               ▼               ▼
     latest pointer cand   action target distribution
               │               │
               ▼               │
        ┌──────────────┐       │
        │ corrector    │◄──────┘
        └──────┬───────┘
               ▼
      posterior pointer state
               │
               ▼
        ┌──────────────┐
        │ controller   │ -> vx, vy, contact/up/down
        └──────┬───────┘
               ▼
           HID emitter
```

task / verification loop:

```text
 recent graph frontier + branches + task text + historic anchors
                              │
                              ▼
                     query aperture assembly
                              │
                              ▼
                        semantic deliberator
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
           target selection             action intent
                 │                          │
                 └────────────┬─────────────┘
                              ▼
                         action compiler
                              │
                              ▼
                          action chunk
                              │
                              ▼
                         pointer servo
                              │
                              ▼
                        new observations
                              │
                              ▼
                           verifier
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
        branch update     chunk success     recovery trigger
```

======================================================================
24. implementation phases
=========================

phase 0: substrate

* event journal
* deterministic replay
* graph storage
* worker orchestration

phase 1: perception MVP

* OCR
* proposals
* layout hints
* pointer candidates
* delta motion

phase 2: belief graph MVP

* entity creation/revision
* text attachment
* affordance role priors
* basic resident set

phase 3: pointer servo MVP

* predictor
* corrector
* velocity controller
* HID emitter integration

phase 4: action compiler + verifier MVP

* tap and scroll chunks
* expectation generation
* simple verifier

phase 5: task deliberator MVP

* query pool assembly
* node ranking
* action intent
* chunk launch

phase 6: recovery + scheduler

* trigger collection
* rule-based recovery
* multi-rate scheduling

phase 7: historical reflection

* historic anchors
* analogy matching
* value-of-retrospection
* reflection outputs into task/recovery loops

phase 8: training and improvement

* teacher-student structure distillation
* verifier training
* retrieval tuning
* control residual learning
* scheduler policy learning

======================================================================
25. what to build first this week
=================================

if i were doing this for real and wanted fastest path to a functioning on-device brain:

1. build the event journal and replay tools first
2. implement pointer posterior + HID velocity loop second
3. implement frame -> OCR / regions / pointer candidates third
4. implement a minimal belief graph fourth
5. implement tap/scroll action chunks fifth
6. implement verifier sixth
7. implement task node ranking seventh
8. add recovery eighth
9. add resident memory manager ninth
10. add historical reflection last

because without replay + pointer + verifier, everything else becomes fake.

======================================================================
26. the sharpest final architecture definition
==============================================

the concrete implementation should be:

an on-device event-sourced control system whose durable truth is an append-only journal plus a revisable typed belief graph; whose fast path is a hidden-state pointer servo and verifier loop; whose slow path is a task-conditioned deliberator over budgeted query apertures assembled from a resident subgraph; whose action semantics are explicit branch-generating chunks; whose memory efficiency comes from residency and accessibility control rather than truth destruction; and whose historical reflection is permitted whenever its expected instrumental value exceeds its delay and compute cost.

that is the serious concrete plan.

the next thing i’d do is write the actual repo layout and interface files: `events.py`, `graph.py`, `scheduler.py`, `pointer.py`, `compiler.py`, `verifier.py`, `recovery.py`, `workspace.py`, `models.py`, `runtime.py`, with exact function signatures and a toy end-to-end loop.
