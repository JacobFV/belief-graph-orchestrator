
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Type

from .belief import BeliefWorker
from .compiler import ActionCompiler
from .graph import BeliefGraph
from .io_streams import StreamingIOManager
from .target import GUITarget
from .journal import EventJournal
from .memory import ResidencyManager
from .models import BrainModels
from .perception import PerceptionWorker
from .pointer import PointerServoWorker, current_phase
from .recovery import RecoveryWorker
from .reflection import HistoricalReflectionWorker
from .scheduler import Scheduler, order_runnable
from .schemas import (
    ComplexState,
    DeltaFeatures,
    GestureState,
    InteractionState,
    PointerPosterior,
    QueryBudget,
    ScaleBand,
    ServoState,
    SubtaskState,
    TaskState,
    TemporalStance,
    VerificationState,
)
from .task import TaskWorker
from .utils import now_ns
from .verifier import VerifierWorker
from .serialization import bundle_from_runtime_state, save_session_bundle


class FrontierVisualWorker:
    def tick(self, state: "RuntimeState") -> None:
        if state.latest_delta is None:
            return
        # use delta as immediate ambiguity / progress signal
        state.ambiguity_score = max(0.0, min(1.0, 0.8 - 3.0 * state.latest_delta.global_change_score))
        state.task_state.progress_estimate = min(1.0, state.task_state.progress_estimate + 0.05 * state.latest_delta.global_change_score)


class BackgroundWorker:
    def __init__(self) -> None:
        self._last_verifier_event_id = 0

    def tick(self, state: "RuntimeState") -> None:
        state.residency.update_scores(state.graph, state)

        # derive branch entropy
        ps = [state.branches[bid].posterior for bid in state.live_branch_ids if bid in state.branches]
        if ps:
            import math
            z = sum(max(p, 1e-6) for p in ps)
            ps = [max(p, 1e-6) / z for p in ps]
            state.branch_entropy = -sum(p * math.log(p + 1e-8) for p in ps) / max(math.log(len(ps) + 1e-8), 1e-6)
        else:
            state.branch_entropy = 0.0

        # create historic anchors / failure patterns from verifier judgments
        for ev in state.event_journal.tail(8, {"verifier_judgment"}):
            if ev.id <= self._last_verifier_event_id:
                continue
            verdict = ev.payload["verdict"]
            label = verdict.label if hasattr(verdict, "label") else verdict["label"]
            kind = "historic_anchor" if label in {"success", "partial"} else "failure_pattern"
            text = f"{label}:{state.task_state.active_goal}:{state.latest_screen_id}"
            node = state.graph.create_node(
                scale=5,
                kind=kind,  # type: ignore[arg-type]
                confidence=0.7,
                state={
                    "summary_text": text,
                    "utility_score": 0.8 if kind == "historic_anchor" else 0.6,
                },
                support_event_ids=[ev.id],
            )
            state.residency.ensure(node.id, state.now_ns)
            self._last_verifier_event_id = ev.id


@dataclass
class RuntimeState:
    session_id: str
    episode_id: str
    body: Any  # GUITarget — typed as Any to avoid circular import issues
    event_journal: EventJournal
    graph: BeliefGraph
    residency: ResidencyManager
    models: BrainModels
    compiler: ActionCompiler

    complexes: dict[str, ComplexState]

    task_state: TaskState = field(default_factory=TaskState)
    subtask_state: SubtaskState = field(default_factory=SubtaskState)
    interaction_state: InteractionState = field(default_factory=InteractionState)
    gesture_state: GestureState = field(default_factory=GestureState)
    servo_state: ServoState = field(default_factory=ServoState)

    pointer: PointerPosterior = field(default_factory=lambda: PointerPosterior(
        x_hat=160.0, y_hat=320.0, vx_hat=0.0, vy_hat=0.0,
        cov=[[10.0,0,0,0],[0,10.0,0,0],[0,0,10.0,0],[0,0,0,10.0]],
        visible_conf=0.0, last_obs_event_id=None, dynamics={"gain_x": 1.0, "gain_y": 1.0}
    ))
    latest_pointer_candidate: Optional[Any] = None
    last_hid_cmd: dict[str, float] = field(default_factory=lambda: {"vx": 0.0, "vy": 0.0})

    active_action_chunk: Optional[Any] = None
    pending_target: Optional[Any] = None
    pending_intent: Optional[Any] = None
    current_subgoal: Optional[Any] = None

    expectations: dict[int, Any] = field(default_factory=dict)
    branches: dict[int, Any] = field(default_factory=dict)
    live_branch_ids: list[int] = field(default_factory=list)
    chunk_root_event_ids: dict[str, int] = field(default_factory=dict)

    latest_delta: Optional[DeltaFeatures] = None
    latest_screen_id: Optional[str] = None
    previous_screen_id: Optional[str] = None

    failure_anchor_node_ids: list[int] = field(default_factory=list)
    recovery_reasons: list[str] = field(default_factory=list)
    force_reflection: bool = False

    task_embedding: list[float] = field(default_factory=lambda: [0.0] * 128)
    _last_embedded_task_text: str = ""

    # ── global workspace vectors (from spec: z_world, z_task, z_motor, z_focus, z_error) ──
    # These are compact summary vectors updated each tick from the graph
    # and state.  Complexes condition their query reads on these.
    z_world: list[float] = field(default_factory=lambda: [0.0] * 64)
    z_motor: list[float] = field(default_factory=lambda: [0.0] * 64)
    z_focus: list[float] = field(default_factory=lambda: [0.0] * 64)
    z_error: list[float] = field(default_factory=lambda: [0.0] * 64)

    verification_state: VerificationState = field(default_factory=lambda: VerificationState(
        micro_ok=1.0, servo_ok=1.0, gesture_ok=1.0,
        interaction_ok=1.0, subtask_ok=1.0, task_ok=1.0,
    ))

    pointer_uncertainty: float = 0.0
    branch_entropy: float = 0.0
    fragile_action_phase: float = 0.0
    pending_timeout_pressure: float = 0.0
    failure_density: float = 0.0
    ambiguity_score: float = 0.0
    analogy_match_score: float = 0.0

    now_ns: int = 0


def default_complexes() -> dict[str, ComplexState]:
    return {
        "pointer": ComplexState(
            id="pointer",
            kind="pointer_servo",
            base_tick_hz=60.0,
            active_scale_band=ScaleBand({0: 0.7, 1: 0.3}, (5, 120), 1, 1),
            stance=TemporalStance(1.0, 0.0, 0.05, 0.0),
            budget=QueryBudget(8, 8, 2, 1, 2, 0, 0),
            energy=1.0,
            lock_strength=0.95,
            anchor_node_ids=[],
        ),
        "action_exec": ComplexState(
            id="action_exec",
            kind="action_execution",
            base_tick_hz=30.0,
            active_scale_band=ScaleBand({1: 0.4, 2: 0.6}, (20, 400), 2, 1),
            stance=TemporalStance(1.0, 0.0, 0.05, 0.0),
            budget=QueryBudget(6, 8, 2, 2, 2, 0, 0),
            energy=1.0,
            lock_strength=0.9,
            anchor_node_ids=[],
        ),
        "frontier_visual": ComplexState(
            id="frontier_visual",
            kind="frontier_visual",
            base_tick_hz=15.0,
            active_scale_band=ScaleBand({1: 0.2, 2: 0.5, 3: 0.3}, (30, 800), 3, 2),
            stance=TemporalStance(0.9, 0.1, 0.1, 0.2),
            budget=QueryBudget(10, 10, 2, 2, 2, 0, 0),
            energy=1.0,
            lock_strength=0.7,
            anchor_node_ids=[],
        ),
        "verifier": ComplexState(
            id="verifier",
            kind="verifier",
            base_tick_hz=12.0,
            active_scale_band=ScaleBand({1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2, 5: 0.1}, (40, 1500), 4, 4),
            stance=TemporalStance(0.8, 0.2, 0.3, 0.0),
            budget=QueryBudget(12, 12, 8, 8, 8, 2, 2),
            energy=1.0,
            lock_strength=0.8,
            anchor_node_ids=[],
        ),
        "task": ComplexState(
            id="task",
            kind="task_pursuit",
            base_tick_hz=3.0,
            active_scale_band=ScaleBand({3: 0.2, 4: 0.45, 5: 0.35}, (500, 30000), 8, 6),
            stance=TemporalStance(0.7, 0.35, 0.25, 0.1),
            budget=QueryBudget(16, 16, 8, 8, 16, 6, 4),
            energy=1.0,
            lock_strength=0.6,
            anchor_node_ids=[],
        ),
        "recovery": ComplexState(
            id="recovery",
            kind="recovery",
            base_tick_hz=4.0,
            active_scale_band=ScaleBand({2: 0.2, 3: 0.3, 4: 0.3, 5: 0.2}, (100, 10000), 8, 8),
            stance=TemporalStance(0.6, 0.5, 0.4, 0.0),
            budget=QueryBudget(8, 8, 10, 8, 8, 8, 4),
            energy=1.0,
            lock_strength=0.5,
            anchor_node_ids=[],
        ),
        "historical_reflection": ComplexState(
            id="historical_reflection",
            kind="historical_reflection",
            base_tick_hz=1.0,
            active_scale_band=ScaleBand({3: 0.1, 4: 0.4, 5: 0.5}, (500, 60000), 12, 10),
            stance=TemporalStance(0.2, 0.9, 0.5, 0.1),
            budget=QueryBudget(4, 4, 8, 4, 10, 16, 12),
            energy=0.5,
            lock_strength=0.3,
            anchor_node_ids=[],
        ),
        "background": ComplexState(
            id="background",
            kind="background",
            base_tick_hz=1.0,
            active_scale_band=ScaleBand({5: 1.0}, (1000, 60000), 20, 10),
            stance=TemporalStance(0.1, 0.7, 0.2, 0.0),
            budget=QueryBudget(0, 0, 0, 0, 0, 0, 0),
            energy=0.3,
            lock_strength=0.1,
            anchor_node_ids=[],
        ),
    }


class Brain:
    def __init__(
        self,
        target_key: str = "",
        target_cls: Type[GUITarget] | None = None,
        use_metadata_hints: bool = True,
        target_instance: GUITarget | None = None,
        # ── backward-compatible aliases ──
        iphone_key: str = "",
        iphone_cls: Type[GUITarget] | None = None,
        iphone_instance: GUITarget | None = None,
    ) -> None:
        key = target_key or iphone_key or "default"
        instance = target_instance or iphone_instance
        cls = target_cls or iphone_cls

        if instance is not None:
            self.body = instance
        elif cls is not None:
            self.body = cls(key)
        else:
            from .backends.mock import MockPhone
            self.body = MockPhone(key)

        self.journal = EventJournal()
        self.graph = BeliefGraph()
        self.residency = ResidencyManager()
        self.models = BrainModels().eval()
        self.compiler = ActionCompiler()

        complexes = default_complexes()

        # ── adapt scheduling based on target capabilities ──
        # Note: pointer must tick fast enough to catch every phase (even short
        # ones like click at 30ms), so we keep it at 60Hz.  The per-tick cost
        # on desktop is negligible (just a cursor readout).
        if self.body.supports_absolute_move:
            # Action execution is faster (no multi-phase velocity approach)
            complexes["action_exec"].base_tick_hz = 15.0

        # Center pointer at body dimensions
        initial_pointer = PointerPosterior(
            x_hat=self.body.width / 2.0,
            y_hat=self.body.height / 2.0,
            vx_hat=0.0, vy_hat=0.0,
            cov=[[10.0,0,0,0],[0,10.0,0,0],[0,0,10.0,0],[0,0,0,10.0]],
            visible_conf=1.0 if self.body.has_direct_cursor else 0.0,
            last_obs_event_id=None,
            dynamics={"gain_x": 1.0, "gain_y": 1.0},
        )

        self.state = RuntimeState(
            session_id="session_1",
            episode_id="episode_1",
            body=self.body,
            event_journal=self.journal,
            graph=self.graph,
            residency=self.residency,
            models=self.models,
            compiler=self.compiler,
            complexes=complexes,
            pointer=initial_pointer,
        )

        desktop_mode = self.body.has_direct_cursor
        self.perception_worker = PerceptionWorker(
            self.models,
            use_metadata_hints=use_metadata_hints,
            desktop_mode=desktop_mode,
            body=self.body,
        )
        self.belief_worker = BeliefWorker()
        self.pointer_worker = PointerServoWorker()
        self.verifier_worker = VerifierWorker()
        self.task_worker = TaskWorker(self.compiler)
        self.recovery_worker = RecoveryWorker()
        self.reflection_worker = HistoricalReflectionWorker()
        self.frontier_worker = FrontierVisualWorker()
        self.background_worker = BackgroundWorker()
        self.scheduler = Scheduler()
        self.io = StreamingIOManager()

        self._last_step_ns = now_ns()

    def poll_body_and_append_events(self) -> list:
        out = []

        # ── streaming I/O (text + audio input, text + speech output) ──
        io_events = self.io.poll_inputs(
            self.body, self.journal,
            self.state.session_id, self.state.episode_id,
            self.state.now_ns,
        )
        for ev in io_events:
            if ev.type == "task_instruction":
                self.state.task_state.active_goal = ev.payload.get("text", "")
        out.extend(io_events)

        # ── legacy single-shot task instruction ──
        task = self.body.get_task_instruction()
        if task:
            ev = self.journal.make_event(
                "task_instruction",
                self.state.session_id,
                self.state.episode_id,
                {"text": task, "source": "poll"},
            )
            self.journal.append(ev)
            self.state.task_state.active_goal = task
            out.append(ev)

        ack = self.body.get_hid_ack()
        if ack:
            ev = self.journal.make_event(
                "hid_ack",
                self.state.session_id,
                self.state.episode_id,
                {"ack": ack},
            )
            self.journal.append(ev)
            out.append(ev)

        fp = self.body.get_new_frame()
        if fp is not None:
            prev = self.state.latest_screen_id
            cur = fp.metadata.get("screen_id")
            self.state.previous_screen_id = prev
            self.state.latest_screen_id = cur
            ev = self.journal.make_event(
                "frame",
                self.state.session_id,
                self.state.episode_id,
                {"image": fp.image, "metadata": fp.metadata, "frame_packet": fp},
                t_capture_ns=fp.t_capture_ns,
            )
            self.journal.append(ev)
            out.append(ev)

        return out

    def maybe_run_perception(self, new_events: list) -> list:
        out = []
        for ev in new_events:
            if ev.type == "frame":
                frame_packet = ev.payload["frame_packet"]
                out.extend(self.perception_worker.process_frame(ev, frame_packet, self.journal))
        for ev in out:
            self.journal.append(ev)
        return out

    def run_complex_at_scale(self, cid: str, lvl: int, dt_s: float) -> None:
        self.state.complexes[cid].last_tick_ns = self.state.now_ns

        if cid == "pointer":
            self.pointer_worker.tick(self.state, dt_s)
        elif cid == "action_exec":
            self._action_exec_tick()
        elif cid == "frontier_visual":
            self.frontier_worker.tick(self.state)
        elif cid == "verifier":
            self.verifier_worker.tick(self.state)
        elif cid == "task":
            self.task_worker.tick(self.state, scale_level=lvl)
        elif cid == "recovery":
            self.recovery_worker.tick(self.state)
        elif cid == "historical_reflection":
            self.reflection_worker.tick(self.state, scale_level=lvl)
        elif cid == "background":
            self.background_worker.tick(self.state)


    def emit_metric_snapshot(self) -> None:
        ev = self.journal.make_event(
            "metric",
            self.state.session_id,
            self.state.episode_id,
            {
                "task_text": self.state.task_state.active_goal,
                "screen_id": self.state.latest_screen_id,
                "pointer_uncertainty": float(self.state.pointer_uncertainty),
                "branch_entropy": float(self.state.branch_entropy),
                "fragile_action_phase": float(self.state.fragile_action_phase),
                "pending_timeout_pressure": float(self.state.pending_timeout_pressure),
                "failure_density": float(self.state.failure_density),
                "ambiguity_score": float(self.state.ambiguity_score),
                "analogy_match_score": float(self.state.analogy_match_score),
                "active_chunk_kind": None if self.state.active_action_chunk is None else self.state.active_action_chunk.kind,
                "current_subgoal": None if self.state.current_subgoal is None else self.state.current_subgoal.description,
                "candidate_target_ids": list(self.state.interaction_state.candidate_target_ids),
                "num_events": len(self.journal.events),
                "num_nodes": len(self.graph.nodes),
            },
        )
        self.journal.append(ev)

    def _action_exec_tick(self) -> None:
        chunk = self.state.active_action_chunk
        if chunk is None:
            self.state.fragile_action_phase = 0.0
            return
        phase_name, _ = current_phase(chunk, self.state.now_ns)
        self.state.gesture_state.current_phase = phase_name
        self.state.gesture_state.active_chunk_id = chunk.id

        # ── fragile-phase scoring ──
        if phase_name in {"contact", "drag", "release", "click"}:
            self.state.fragile_action_phase = 0.9
        elif phase_name in {"slow_approach", "dwell", "move_to"}:
            self.state.fragile_action_phase = 0.5
        elif phase_name in {"type_text", "key_combo", "scroll_wheel"}:
            self.state.fragile_action_phase = 0.6
        else:
            self.state.fragile_action_phase = 0.2

        elapsed_ms = (self.state.now_ns - chunk.started_ns) / 1e6
        self.state.pending_timeout_pressure = min(1.0, elapsed_ms / max(chunk.timeout_ms, 1))

    def _update_workspace_vectors(self) -> None:
        """
        Update the global workspace summary vectors from current state.
        These are the z_world, z_motor, z_focus, z_error vectors that
        complexes use to condition their query reads.
        """
        s = self.state
        pp = s.pointer

        # z_world: summarize the belief graph state
        # Mean of resident node z_obj embeddings (if any)
        n_resident = 0
        z_world_acc = [0.0] * 64
        for nid in list(s.residency.resident_ids)[:100]:  # cap for speed
            node = self.graph.node(nid)
            if node.z_obj:
                for i in range(min(64, len(node.z_obj))):
                    z_world_acc[i] += node.z_obj[i]
                n_resident += 1
        if n_resident > 0:
            s.z_world = [v / n_resident for v in z_world_acc]

        # z_motor: pointer state + control state
        s.z_motor = [
            pp.x_hat / max(self.body.width, 1), pp.y_hat / max(self.body.height, 1),
            pp.vx_hat, pp.vy_hat,
            pp.visible_conf, s.pointer_uncertainty,
            s.fragile_action_phase, s.pending_timeout_pressure,
        ] + [0.0] * 56

        # z_focus: current target / subgoal
        z_focus = [0.0] * 64
        if s.pending_target and s.pending_target.node_ids:
            nid = s.pending_target.node_ids[0]
            if nid in self.graph.nodes:
                n = self.graph.node(nid)
                if n.z_obj:
                    z_focus = n.z_obj[:64] + [0.0] * max(0, 64 - len(n.z_obj))
        s.z_focus = z_focus

        # z_error: failure / recovery state
        s.z_error = [
            s.failure_density, s.branch_entropy, s.ambiguity_score,
            s.analogy_match_score,
            s.verification_state.micro_ok, s.verification_state.servo_ok,
            s.verification_state.gesture_ok, s.verification_state.interaction_ok,
            s.verification_state.subtask_ok, s.verification_state.task_ok,
        ] + [0.0] * 54

    def _propagate_contracts(self) -> None:
        """
        Cross-scale contract propagation.

        Top-down: task → subtask → interaction → gesture → servo
          Each level constrains the one below.

        Bottom-up: servo → gesture → interaction → subtask → task
          Each level signals status upward.
        """
        s = self.state

        # ── top-down: task constrains subtask ──
        if s.current_subgoal:
            s.subtask_state.active_route = s.current_subgoal.description
            if s.task_state.risk_posture == "conservative":
                s.subtask_state.contradiction_score = max(s.subtask_state.contradiction_score, 0.3)

        # ── top-down: subtask constrains interaction ──
        if s.subtask_state.target_region_ids:
            for nid in s.subtask_state.target_region_ids:
                if nid not in s.interaction_state.candidate_target_ids:
                    s.interaction_state.candidate_target_ids.append(nid)

        # ── top-down: interaction constrains gesture ──
        if s.pending_target and s.active_action_chunk:
            s.gesture_state.timeout_pressure = s.pending_timeout_pressure

        # ── top-down: gesture constrains servo ──
        if s.active_action_chunk:
            pp = s.pointer
            target = s.active_action_chunk.target_distribution.get("mean")
            if target:
                from .utils import l2
                err = l2((pp.x_hat, pp.y_hat), target)
                s.servo_state.current_error_xy = (target[0] - pp.x_hat, target[1] - pp.y_hat)
            s.servo_state.visibility_confidence = pp.visible_conf

        # ── bottom-up: servo signals upward ──
        if s.pointer_uncertainty > 0.6:
            s.gesture_state.timeout_pressure = min(1.0, s.gesture_state.timeout_pressure + 0.1)
            s.interaction_state.ambiguity_score = min(1.0, s.interaction_state.ambiguity_score + 0.05)

        # ── bottom-up: verification state signals upward ──
        vs = s.verification_state
        if vs.gesture_ok < 0.4:
            s.interaction_state.ambiguity_score = min(1.0, s.interaction_state.ambiguity_score + 0.1)
        if vs.interaction_ok < 0.4:
            s.subtask_state.contradiction_score = min(1.0, s.subtask_state.contradiction_score + 0.1)
        if vs.subtask_ok < 0.4:
            s.task_state.risk_posture = "conservative"
        elif vs.subtask_ok > 0.8:
            s.task_state.risk_posture = "normal"

    def step(self) -> None:
        self.state.now_ns = now_ns()
        dt_s = max(1e-3, (self.state.now_ns - self._last_step_ns) / 1e9)
        self._last_step_ns = self.state.now_ns

        new_events = self.poll_body_and_append_events()
        perception_events = self.maybe_run_perception(new_events)
        self.belief_worker.step(self.state, new_events + perception_events)

        runnable = self.scheduler.step(self.state, self.state.now_ns)
        for cid, lvl in order_runnable(runnable):
            self.run_complex_at_scale(cid, lvl, dt_s)

        # Update global workspace vectors + cross-scale contract propagation
        self._update_workspace_vectors()
        self._propagate_contracts()

        # Always allow low-cost maintenance of residency.
        self.background_worker.tick(self.state)

        # Flush any pending text/speech output to the body
        self.io.flush_outputs(
            self.body, self.journal,
            self.state.session_id, self.state.episode_id,
            self.state.now_ns,
        )

        self.emit_metric_snapshot()

    def run(self, steps: int = 200) -> None:
        for _ in range(steps):
            self.step()

    def bundle(self) -> dict[str, Any]:
        return bundle_from_runtime_state(self.state)

    def save_bundle(self, path: str) -> None:
        save_session_bundle(self.bundle(), path)

    def say(self, text: str) -> None:
        """Emit a text response to the user."""
        self.io.output.emit_text(text, self.state.now_ns)

    def speak(self, text: str, **kwargs) -> None:
        """Emit a speech output to the user."""
        self.io.output.emit_speech(text, self.state.now_ns, **kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.state.task_state.active_goal,
            "screen": self.state.latest_screen_id,
            "pointer": (self.state.pointer.x_hat, self.state.pointer.y_hat),
            "active_chunk": None if self.state.active_action_chunk is None else self.state.active_action_chunk.kind,
            "subgoal": None if self.state.current_subgoal is None else self.state.current_subgoal.description,
            "num_events": len(self.journal.events),
            "num_nodes": len(self.graph.nodes),
            "branches": {bid: vars(b) for bid, b in self.state.branches.items()},
        }
