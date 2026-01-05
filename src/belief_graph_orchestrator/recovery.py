
from __future__ import annotations

from .schemas import ActionIntent, TargetSelection
from .utils import bbox_center


def collect_recovery_triggers(state: "RuntimeState") -> list[str]:
    triggers = list(state.recovery_reasons)
    state.recovery_reasons.clear()

    if state.pointer_uncertainty > 0.8:
        triggers.append("pointer_divergence")
    if state.active_action_chunk is not None:
        elapsed_ms = (state.now_ns - state.active_action_chunk.started_ns) / 1e6
        if elapsed_ms > state.active_action_chunk.timeout_ms:
            triggers.append("timeout")
    if state.failure_density > 0.7:
        triggers.append("repeated_failure")
    if state.branch_entropy > 0.8:
        triggers.append("ambiguous_branch")

    # ── multi-scale failure detection ──
    vs = getattr(state, "verification_state", None)
    if vs is not None:
        if vs.micro_ok < 0.3:
            triggers.append("motor_failure_L0")
        if vs.servo_ok < 0.3:
            triggers.append("servo_failure_L1")
        if vs.gesture_ok < 0.3 and vs.servo_ok > 0.5:
            triggers.append("gesture_failure_L2")
        if vs.subtask_ok < 0.3 and vs.interaction_ok > 0.5:
            triggers.append("route_failure_L4")

    # dedupe
    out = []
    for t in triggers:
        if t not in out:
            out.append(t)
    return out


class RecoveryWorker:
    def choose_recovery_plan(self, triggers: list[str], state: "RuntimeState") -> dict:
        # ── scale-aware recovery: match recovery strategy to failure level ──

        # L0/L1: motor / pointer problems → tighten control
        if "pointer_divergence" in triggers or "motor_failure_L0" in triggers or "servo_failure_L1" in triggers:
            if state.body.has_direct_cursor:
                return {"kind": "pause_and_wait", "scale_jump": 1}
            return {"kind": "reacquire_pointer", "scale_jump": 0}

        # L2: gesture failed but servo was fine → retry the gesture
        if "gesture_failure_L2" in triggers:
            return {"kind": "pause_and_wait", "scale_jump": 2}

        # L4: route seems wrong → back up and rescan at broader scale
        if "route_failure_L4" in triggers:
            if state.body.supports_keyboard:
                return {"kind": "press_escape_then_rescan", "scale_jump": 4}
            return {"kind": "back_then_rescan", "scale_jump": 4}

        # Scale-annotated verification failures → jump to that scale
        for trigger in triggers:
            if trigger.startswith("verification_failure_L"):
                try:
                    lvl = int(trigger.split("_L")[1])
                    if lvl <= 1:
                        return {"kind": "reacquire_pointer", "scale_jump": lvl}
                    if lvl == 2:
                        return {"kind": "pause_and_wait", "scale_jump": 2}
                    if lvl >= 3:
                        if state.body.supports_keyboard:
                            return {"kind": "press_escape_then_rescan", "scale_jump": lvl}
                        return {"kind": "back_then_rescan", "scale_jump": lvl}
                except (ValueError, IndexError):
                    pass

        # Generic triggers
        if "timeout" in triggers:
            if state.body.supports_keyboard:
                return {"kind": "press_escape", "scale_jump": 3}
            return {"kind": "pause_and_wait", "scale_jump": 2}
        if "ambiguous_branch" in triggers:
            return {"kind": "invoke_historical_reflection", "scale_jump": 5}
        if "repeated_failure" in triggers:
            if state.body.supports_keyboard:
                return {"kind": "press_escape_then_rescan", "scale_jump": 4}
            return {"kind": "back_then_rescan", "scale_jump": 4}
        return {"kind": "pause_and_wait", "scale_jump": 2}

    def enact_recovery_plan(self, plan: dict, state: "RuntimeState") -> None:
        kind = plan["kind"]
        if kind == "reacquire_pointer":
            state.active_action_chunk = None
            state.pointer.visible_conf = 0.0
            return
        if kind == "pause_and_wait":
            state.active_action_chunk = state.compiler.compile_wait(state)
            state.gesture_state.active_chunk_id = state.active_action_chunk.id
            return
        if kind == "back_then_rescan":
            state.active_action_chunk = state.compiler.compile_back(state)
            state.gesture_state.active_chunk_id = state.active_action_chunk.id
            return
        if kind == "invoke_historical_reflection":
            state.force_reflection = True
            return
        if kind == "press_escape":
            state.active_action_chunk = state.compiler.compile_key_combo("Escape", [], state)
            state.gesture_state.active_chunk_id = state.active_action_chunk.id
            return
        if kind == "press_escape_then_rescan":
            state.active_action_chunk = state.compiler.compile_key_combo("Escape", [], state)
            state.gesture_state.active_chunk_id = state.active_action_chunk.id
            state.failure_density = max(0.0, state.failure_density - 0.1)
            return

    def tick(self, state: "RuntimeState") -> None:
        triggers = collect_recovery_triggers(state)
        if not triggers:
            return
        plan = self.choose_recovery_plan(triggers, state)
        ev = state.event_journal.make_event(
            "recovery_trigger",
            state.session_id,
            state.episode_id,
            {"triggers": triggers, "plan": plan},
        )
        state.event_journal.append(ev)
        self.enact_recovery_plan(plan, state)
