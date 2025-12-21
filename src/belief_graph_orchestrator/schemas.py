from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

EventType = Literal[
    "frame", "ocr", "regions", "layout", "pointer_candidates", "delta", "task_instruction",
    "action_issued", "hid_ack", "belief_revision", "branch_update", "verifier_judgment",
    "recovery_trigger", "teacher_structure", "metric",
    # streaming I/O
    "text_input_chunk",     # incremental text from user (typed, dictated, pasted)
    "audio_input_frame",    # raw audio samples from microphone
    "transcript_chunk",     # speech-to-text output (partial or final)
    "speech_output",        # text-to-speech request emitted by the brain
    "text_output",          # text response emitted by the brain
]

NodeKind = Literal[
    "micro_residual", "pointer_posterior", "gesture_chunk", "gesture_phase", "text_span",
    "affordance", "container", "candidate_target", "interaction_outcome", "screen_region",
    "route_hypothesis", "task_subgoal", "goal_state", "historic_anchor", "analogy_anchor",
    "failure_pattern", "episode_summary", "counterfactual_entity", "expectation",
]

EdgeKind = Literal[
    "supports", "contradicts", "contains", "same_entity_as", "caused_by", "predicted_by",
    "resolved_by", "summarizes", "refines", "retrieved_for", "relevant_to", "in_branch",
    "near_spatially", "next_in_time",
]

ComplexKind = Literal[
    "pointer_servo", "action_execution", "frontier_visual", "verifier", "task_pursuit",
    "recovery", "background", "historical_reflection",
]

@dataclass
class FramePacket:
    image: Any
    t_capture_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)

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
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class LayoutHint:
    kind: Literal["nav_bar", "tab_bar", "sheet", "modal", "list", "keyboard", "toolbar", "menubar", "sidebar", "dialog", "statusbar", "unknown"]
    bbox: tuple[float, float, float, float]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PointerCandidate:
    x: float
    y: float
    confidence: float
    signature: list[float]

@dataclass
class DeltaFeatures:
    changed_regions: list[tuple[float, float, float, float]]
    global_change_score: float
    dominant_motion: tuple[float, float]

@dataclass
class BeliefNode:
    id: int
    scale: int
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
class ScaleBand:
    level_probs: dict[int, float]
    temporal_horizon_ms: tuple[int, int]
    graph_radius: int
    branch_width: int

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
    base_tick_hz: float
    active_scale_band: ScaleBand
    stance: TemporalStance
    budget: QueryBudget
    energy: float
    lock_strength: float
    anchor_node_ids: list[int]
    latent: dict[str, Any] = field(default_factory=dict)
    last_tick_ns: int = 0

@dataclass
class TaskState:
    active_goal: str = ""
    success_criteria: list[str] = field(default_factory=list)
    risk_posture: str = "normal"
    progress_estimate: float = 0.0
    z: list[float] = field(default_factory=list)

@dataclass
class SubtaskState:
    active_route: str = ""
    target_region_ids: list[int] = field(default_factory=list)
    expected_transition_types: list[str] = field(default_factory=list)
    contradiction_score: float = 0.0
    z: list[float] = field(default_factory=list)

@dataclass
class InteractionState:
    candidate_target_ids: list[int] = field(default_factory=list)
    target_distribution: dict[str, Any] = field(default_factory=dict)
    expected_feedback: list[int] = field(default_factory=list)
    ambiguity_score: float = 0.0
    z: list[float] = field(default_factory=list)

@dataclass
class GestureState:
    active_chunk_id: Optional[str] = None
    current_phase: Optional[str] = None
    phase_progress: float = 0.0
    timeout_pressure: float = 0.0
    z: list[float] = field(default_factory=list)

@dataclass
class ServoState:
    pointer_posterior_id: int = -1
    current_error_xy: tuple[float, float] = (0.0, 0.0)
    visibility_confidence: float = 0.0
    gain_profile: dict[str, float] = field(default_factory=dict)
    z: list[float] = field(default_factory=list)

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

@dataclass
class ActionPhase:
    name: str
    params: dict[str, Any]

@dataclass
class ActionChunk:
    id: str
    kind: Literal["tap", "scroll", "drag", "type", "wait", "back", "dismiss", "explore", "key_combo", "type_text"]
    target_distribution: dict[str, Any]
    phases: list[ActionPhase]
    velocity_profile: dict[str, Any]
    timeout_ms: int
    expectation_ids: list[int]
    fallback_policy: dict[str, Any]
    target_node_ids: list[int] = field(default_factory=list)
    intent_confidence: float = 0.0
    started_ns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Expectation:
    id: int
    kind: Literal["target_state_change", "overlay_appears", "keyboard_appears", "scroll_displacement", "navigation_change", "text_change", "none_visible"]
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
    kind: Literal["tap", "scroll", "drag", "type", "wait", "back", "dismiss", "explore", "key_combo", "type_text"]
    params: dict[str, Any]
    confidence: float

@dataclass
class VerificationState:
    micro_ok: float
    servo_ok: float
    gesture_ok: float
    interaction_ok: float
    subtask_ok: float
    task_ok: float

@dataclass
class VerifierVerdict:
    label: Literal["success", "partial", "failure", "ambiguous", "delayed"]
    scale_failures: dict[int, float]
    branch_posteriors: dict[int, float]
    notes: dict[str, Any]

@dataclass
class WorkspaceNode:
    node_id: int
    scale: int
    kind: str
    bbox: Optional[tuple[float, float, float, float]]
    text: Optional[str]
    confidence: float
    state: dict[str, Any]
    z_obj: Optional[list[float]]
    z_dyn: Optional[list[float]]
    z_belief: Optional[list[float]]

@dataclass
class Workspace:
    task_state: TaskState
    subtask_state: SubtaskState
    interaction_state: InteractionState
    gesture_state: GestureState
    servo_state: ServoState
    branch_summary: dict[str, Any]
    nodes: list[WorkspaceNode]

@dataclass
class ResidencyRecord:
    node_id: int
    hot_score: float
    last_access_ns: int
    recent_access_count: int
    frontier_score: float
    branch_score: float
    motor_score: float
    historic_score: float
    anomaly_score: float

@dataclass
class HistoricAnchorState:
    summary_text: str
    utility_score: float
    analogy_signature: list[float]
    retrieval_count: int
    last_helped_ns: int
    linked_failure_patterns: list[int]

@dataclass
class TrainingTrace:
    session_id: str
    event_ids: list[int]
    graph_node_ids: list[int]
    action_chunk_ids: list[str]
    verifier_judgments: list[dict]
    outcome_summary: dict[str, Any]
