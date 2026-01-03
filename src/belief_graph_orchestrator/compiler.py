
from __future__ import annotations

from .schemas import ActionChunk, ActionIntent, ActionPhase, Branch, Expectation, TargetSelection
from .utils import new_id, normalize


class ActionCompiler:
    def __init__(self) -> None:
        self._next_expectation_id = 1
        self._next_branch_id = 1

    def next_expectation_id(self) -> int:
        x = self._next_expectation_id
        self._next_expectation_id += 1
        return x

    def next_branch_id(self) -> int:
        x = self._next_branch_id
        self._next_branch_id += 1
        return x

    # ── top-level dispatch ───────────────────────────────────────────

    def compile(self, intent: ActionIntent, target: TargetSelection, state: "RuntimeState") -> ActionChunk:
        if intent.kind == "tap":
            return self.compile_tap(target, state)
        if intent.kind == "scroll":
            return self.compile_scroll(target, intent.params.get("direction", "down"), intent.params.get("amount", 0.5), state)
        if intent.kind == "back":
            return self.compile_back(state)
        if intent.kind == "dismiss":
            return self.compile_dismiss(state)
        if intent.kind == "wait":
            return self.compile_wait(state)
        if intent.kind == "type_text":
            return self.compile_type_text(target, intent.params.get("text", ""), state)
        if intent.kind == "key_combo":
            return self.compile_key_combo(intent.params.get("key", ""), intent.params.get("modifiers", []), state)
        return self.compile_tap(target, state)

    # ── tap ──────────────────────────────────────────────────────────

    def compile_tap(self, target: TargetSelection, state: "RuntimeState") -> ActionChunk:
        if state.body.supports_absolute_move:
            return self._compile_tap_absolute(target, state)
        return self._compile_tap_velocity(target, state)

    def _compile_tap_velocity(self, target: TargetSelection, state: "RuntimeState") -> ActionChunk:
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
                "far_gain": 0.10,
                "near_gain": 0.04,
                "max_vel": 1.0,
                "deadband_px": 3.0,
            },
            timeout_ms=1200,
            expectation_ids=self.create_expectations_for_tap(target, state),
            fallback_policy={"on_fail": "retarget_or_recover"},
            target_node_ids=list(target.node_ids),
            intent_confidence=target.confidence,
            started_ns=state.now_ns,
        )

    def _compile_tap_absolute(self, target: TargetSelection, state: "RuntimeState") -> ActionChunk:
        return ActionChunk(
            id=new_id("tap"),
            kind="tap",
            target_distribution=target.target_distribution,
            phases=[
                ActionPhase("move_to", {"mode": "absolute"}),
                ActionPhase("click", {"button": "left"}),
                ActionPhase("verify", {"window_ms": 300}),
            ],
            velocity_profile={},
            timeout_ms=600,
            expectation_ids=self.create_expectations_for_tap(target, state),
            fallback_policy={"on_fail": "retarget_or_recover"},
            target_node_ids=list(target.node_ids),
            intent_confidence=target.confidence,
            started_ns=state.now_ns,
        )

    # ── scroll ───────────────────────────────────────────────────────

    def compile_scroll(self, target: TargetSelection, direction: str, amount: float, state: "RuntimeState") -> ActionChunk:
        if state.body.supports_absolute_move:
            return self._compile_scroll_absolute(target, direction, amount, state)
        return self._compile_scroll_velocity(target, direction, amount, state)

    def _compile_scroll_velocity(self, target: TargetSelection, direction: str, amount: float, state: "RuntimeState") -> ActionChunk:
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
                "far_gain": 0.10,
                "near_gain": 0.08,
                "drag_vel": 0.55,
                "max_vel": 0.8,
                "deadband_px": 4.0,
            },
            timeout_ms=1800,
            expectation_ids=self.create_expectations_for_scroll(target, direction, amount, state),
            fallback_policy={"on_fail": "rescan_scrollable_container"},
            target_node_ids=list(target.node_ids),
            intent_confidence=target.confidence,
            metadata={"direction": direction, "amount": amount},
            started_ns=state.now_ns,
        )

    def _compile_scroll_absolute(self, target: TargetSelection, direction: str, amount: float, state: "RuntimeState") -> ActionChunk:
        return ActionChunk(
            id=new_id("scroll"),
            kind="scroll",
            target_distribution=target.target_distribution,
            phases=[
                ActionPhase("move_to", {"mode": "absolute"}),
                ActionPhase("scroll_wheel", {"direction": direction, "delta": amount * 5.0}),
                ActionPhase("verify_displacement", {"window_ms": 300}),
            ],
            velocity_profile={},
            timeout_ms=800,
            expectation_ids=self.create_expectations_for_scroll(target, direction, amount, state),
            fallback_policy={"on_fail": "rescan_scrollable_container"},
            target_node_ids=list(target.node_ids),
            intent_confidence=target.confidence,
            metadata={"direction": direction, "amount": amount},
            started_ns=state.now_ns,
        )

    # ── back ─────────────────────────────────────────────────────────

    def compile_back(self, state: "RuntimeState") -> ActionChunk:
        if state.body.supports_keyboard:
            return self.compile_key_combo("ArrowLeft", ["Alt"], state)
        dist = {"mean": (48.0, 40.0), "radius": 10.0}
        return ActionChunk(
            id=new_id("back"),
            kind="back",
            target_distribution=dist,
            phases=[
                ActionPhase("approach", {"mode": "velocity_control"}),
                ActionPhase("contact", {"button": "primary"}),
                ActionPhase("dwell", {"ms": 40}),
                ActionPhase("release", {}),
                ActionPhase("verify", {"window_ms": 500}),
            ],
            velocity_profile={"far_gain": 0.10, "near_gain": 0.04, "max_vel": 1.0, "deadband_px": 3.0},
            timeout_ms=1200,
            expectation_ids=[],
            fallback_policy={"on_fail": "recover"},
            target_node_ids=[],
            started_ns=state.now_ns,
        )

    # ── dismiss ──────────────────────────────────────────────────────

    def compile_dismiss(self, state: "RuntimeState") -> ActionChunk:
        if state.body.supports_keyboard:
            return self.compile_key_combo("Escape", [], state)
        dist = {"mean": (state.pointer.x_hat, state.pointer.y_hat), "radius": 5.0}
        return ActionChunk(
            id=new_id("dismiss"),
            kind="dismiss",
            target_distribution=dist,
            phases=[ActionPhase("release", {}), ActionPhase("verify", {"window_ms": 250})],
            velocity_profile={"far_gain": 0.0, "near_gain": 0.0, "max_vel": 0.0, "deadband_px": 1.0},
            timeout_ms=400,
            expectation_ids=[],
            fallback_policy={"on_fail": "recover"},
            target_node_ids=[],
            started_ns=state.now_ns,
        )

    # ── wait ─────────────────────────────────────────────────────────

    def compile_wait(self, state: "RuntimeState") -> ActionChunk:
        dist = {"mean": (state.pointer.x_hat, state.pointer.y_hat), "radius": 5.0}
        return ActionChunk(
            id=new_id("wait"),
            kind="wait",
            target_distribution=dist,
            phases=[ActionPhase("verify", {"window_ms": 700})],
            velocity_profile={"far_gain": 0.0, "near_gain": 0.0, "max_vel": 0.0, "deadband_px": 1.0},
            timeout_ms=800,
            expectation_ids=[],
            fallback_policy={"on_fail": "recover"},
            target_node_ids=[],
            started_ns=state.now_ns,
        )

    # ── type_text ────────────────────────────────────────────────────

    def compile_type_text(self, target: TargetSelection, text: str, state: "RuntimeState") -> ActionChunk:
        phases = []
        if state.body.supports_absolute_move:
            phases.append(ActionPhase("move_to", {"mode": "absolute"}))
            phases.append(ActionPhase("click", {"button": "left"}))
        else:
            phases.append(ActionPhase("approach", {"mode": "velocity_control"}))
            phases.append(ActionPhase("slow_approach", {"radius_px": 24}))
            phases.append(ActionPhase("contact", {"button": "primary"}))
            phases.append(ActionPhase("dwell", {"ms": 60}))
            phases.append(ActionPhase("release", {}))
        phases.append(ActionPhase("type_text", {"text": text}))
        phases.append(ActionPhase("verify", {"window_ms": 400}))

        return ActionChunk(
            id=new_id("type_text"),
            kind="type_text",
            target_distribution=target.target_distribution,
            phases=phases,
            velocity_profile={"far_gain": 0.10, "near_gain": 0.04, "max_vel": 1.0, "deadband_px": 3.0} if not state.body.supports_absolute_move else {},
            timeout_ms=2000,
            expectation_ids=self._create_expectations_for_type(target, state),
            fallback_policy={"on_fail": "recover"},
            target_node_ids=list(target.node_ids),
            intent_confidence=target.confidence,
            metadata={"text": text},
            started_ns=state.now_ns,
        )

    # ── key_combo ────────────────────────────────────────────────────

    def compile_key_combo(self, key: str, modifiers: list[str], state: "RuntimeState") -> ActionChunk:
        return ActionChunk(
            id=new_id("key_combo"),
            kind="key_combo",
            target_distribution={"mean": (state.pointer.x_hat, state.pointer.y_hat)},
            phases=[
                ActionPhase("key_combo", {"key": key, "modifiers": modifiers}),
                ActionPhase("verify", {"window_ms": 300}),
            ],
            velocity_profile={},
            timeout_ms=600,
            expectation_ids=[],
            fallback_policy={"on_fail": "recover"},
            target_node_ids=[],
            started_ns=state.now_ns,
        )

    # ── expectations ─────────────────────────────────────────────────

    def create_expectations_for_tap(self, target: TargetSelection, state: "RuntimeState") -> list[int]:
        exps = []
        for nid in target.node_ids:
            node = state.graph.node(nid)
            role_probs = node.state.get("role_probs", {})
            candidates = []
            if role_probs.get("ActionableDiscrete", 0.0) > 0.2:
                candidates.append(("target_state_change", 0.35))
                candidates.append(("navigation_change", 0.30))
                candidates.append(("overlay_appears", 0.15))
                candidates.append(("keyboard_appears", 0.10))
                candidates.append(("none_visible", 0.10))
            else:
                candidates.append(("none_visible", 0.8))
                candidates.append(("target_state_change", 0.2))
            total = sum(p for _, p in candidates)
            candidates = [(k, p / total) for k, p in candidates]
            for kind, p in candidates:
                eid = self.next_expectation_id()
                exp = Expectation(id=eid, kind=kind, target_node_ids=[nid], params={}, confidence=p)
                state.expectations[eid] = exp
                exps.append(eid)
        return exps

    def create_expectations_for_scroll(self, target: TargetSelection, direction: str, amount: float, state: "RuntimeState") -> list[int]:
        eid = self.next_expectation_id()
        exp = Expectation(
            id=eid,
            kind="scroll_displacement",
            target_node_ids=target.node_ids,
            params={"direction": direction, "amount": amount},
            confidence=0.9,
        )
        state.expectations[eid] = exp
        return [eid]

    def _create_expectations_for_type(self, target: TargetSelection, state: "RuntimeState") -> list[int]:
        eid = self.next_expectation_id()
        exp = Expectation(
            id=eid,
            kind="text_change",
            target_node_ids=list(target.node_ids),
            params={},
            confidence=0.9,
        )
        state.expectations[eid] = exp
        return [eid]

    # ── branches ─────────────────────────────────────────────────────

    def create_branches_for_action(self, action_event_id: int, chunk: ActionChunk, state: "RuntimeState") -> list[int]:
        by_kind: dict[str, list[int]] = {}
        for eid in chunk.expectation_ids:
            exp = state.expectations[eid]
            by_kind.setdefault(exp.kind, []).append(eid)

        branches = []
        for kind, eids in by_kind.items():
            bid = self.next_branch_id()
            branch = Branch(
                id=bid,
                root_action_event_id=action_event_id,
                prior=1.0,
                posterior=1.0,
                status="live",
                expectation_ids=eids,
                node_ids=[nid for eid in eids for nid in state.expectations[eid].target_node_ids],
            )
            state.branches[bid] = branch
            branches.append(branch)

        priors = normalize([max(1e-6, sum(state.expectations[eid].confidence for eid in b.expectation_ids)) for b in branches])
        for b, p in zip(branches, priors):
            b.prior = p
            b.posterior = p
        return [b.id for b in branches]
