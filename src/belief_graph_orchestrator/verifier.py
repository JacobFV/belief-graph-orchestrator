
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .schemas import VerificationState, VerifierVerdict


def _latest_screen_id(state: "RuntimeState") -> str | None:
    frames = state.event_journal.tail(8, {"frame"})
    if not frames:
        return None
    return frames[-1].payload.get("metadata", {}).get("screen_id")


def _prev_screen_id(state: "RuntimeState") -> str | None:
    frames = state.event_journal.tail(16, {"frame"})
    if len(frames) < 2:
        return None
    return frames[-2].payload.get("metadata", {}).get("screen_id")


# ── per-expectation scoring (unchanged) ──────────────────────────────

def score_target_state_change(exp, state) -> float:
    target_ids = exp.target_node_ids
    if not target_ids:
        return 0.0
    for nid in target_ids:
        if nid in state.graph.nodes:
            node = state.graph.node(nid)
            if node.state.get("selected", False):
                return 0.9
    if state.latest_delta is not None:
        return min(0.9, 5.0 * state.latest_delta.global_change_score)
    return 0.1


def score_overlay_appears(exp, state) -> float:
    for nid in state.graph.nodes_by_kind("container"):
        node = state.graph.node(nid)
        kind = node.state.get("layout_kind")
        if kind in {"modal", "sheet", "keyboard"}:
            return 0.8
    return 0.0


def score_keyboard_appears(exp, state) -> float:
    if _latest_screen_id(state) == "compose":
        return 0.4
    return 0.0


def score_scroll_displacement(exp, state) -> float:
    if state.latest_delta is None:
        return 0.0
    return min(0.95, 8.0 * state.latest_delta.global_change_score)


def score_navigation_change(exp, state) -> float:
    cur = _latest_screen_id(state)
    prev = _prev_screen_id(state)
    if cur is not None and prev is not None and cur != prev:
        return 0.98
    if state.latest_delta is not None:
        return min(0.6, 3.5 * state.latest_delta.global_change_score)
    return 0.0


def score_text_change(exp, state) -> float:
    for nid in exp.target_node_ids:
        if nid in state.graph.nodes:
            node = state.graph.node(nid)
            if node.state.get("role_probs", {}).get("TextEntry", 0.0) > 0.5:
                return 0.7
    if state.latest_delta is not None and state.latest_delta.global_change_score > 0.1:
        return 0.5
    return 0.1


def score_none_visible(exp, state) -> float:
    if state.latest_delta is None:
        return 0.8
    return max(0.0, 1.0 - 8.0 * state.latest_delta.global_change_score)


def score_expectation(exp, state) -> float:
    if exp.kind == "target_state_change":
        return score_target_state_change(exp, state)
    if exp.kind == "overlay_appears":
        return score_overlay_appears(exp, state)
    if exp.kind == "keyboard_appears":
        return score_keyboard_appears(exp, state)
    if exp.kind == "scroll_displacement":
        return score_scroll_displacement(exp, state)
    if exp.kind == "navigation_change":
        return score_navigation_change(exp, state)
    if exp.kind == "text_change":
        return score_text_change(exp, state)
    if exp.kind == "none_visible":
        return score_none_visible(exp, state)
    return 0.0


# ── multi-scale verification ────────────────────────────────────────
#
# The spec defines V_t = {V_0, V_1, ..., V_5} — per-scale verdicts.
# The system must distinguish motor failure from targeting failure from
# route failure.  Each scale looks at different evidence:
#
#   L0/L1 (micro/servo):  pointer motion, local visual feedback
#   L2    (gesture):      gesture phase completed, contact landed
#   L3    (interaction):  target node state changed, expected feedback
#   L4    (subtask):      screen transition, route progress
#   L5    (task):         overall task progress

def _score_micro_servo(state: "RuntimeState") -> float:
    """L0-L1: Did the pointer reach the target and is control stable?"""
    if state.pointer_uncertainty > 0.5:
        return 0.3  # pointer drifted
    # Check if pointer is near the target
    chunk = state.active_action_chunk
    if chunk and "mean" in chunk.target_distribution:
        tx, ty = chunk.target_distribution["mean"]
        from .utils import l2
        d = l2((state.pointer.x_hat, state.pointer.y_hat), (tx, ty))
        return min(1.0, max(0.2, 1.0 - d / 100.0))
    return 0.7


def _score_gesture(state: "RuntimeState") -> float:
    """L2: Did the gesture complete its phase sequence?"""
    phase = state.gesture_state.current_phase
    if phase in {"verify", "verify_displacement"}:
        return 0.8  # reached verify phase → gesture executed
    if phase in {"release"}:
        return 0.6
    return 0.3


def _score_interaction(state: "RuntimeState", best_branch_score: float) -> float:
    """L3: Did the target respond as expected?"""
    return min(1.0, best_branch_score * 1.2)


def _score_subtask(state: "RuntimeState") -> float:
    """L4: Did we make subtask/route progress?"""
    cur = _latest_screen_id(state)
    prev = state.previous_screen_id
    if cur is not None and prev is not None and cur != prev:
        return 0.9  # screen transition → strong subtask signal
    if state.latest_delta is not None and state.latest_delta.global_change_score > 0.15:
        return 0.6
    return 0.3


def _score_task(state: "RuntimeState") -> float:
    """L5: Overall task progress."""
    return min(1.0, state.task_state.progress_estimate + 0.1)


def compute_verification_state(state: "RuntimeState", best_branch_score: float) -> VerificationState:
    """Compute per-scale verification scores."""
    return VerificationState(
        micro_ok=_score_micro_servo(state),
        servo_ok=_score_micro_servo(state),
        gesture_ok=_score_gesture(state),
        interaction_ok=_score_interaction(state, best_branch_score),
        subtask_ok=_score_subtask(state),
        task_ok=_score_task(state),
    )


# ── phase gating ─────────────────────────────────────────────────────

# Phases where the action is still being performed — the verifier must
# wait for the verify/verify_displacement phase before scoring outcomes.
_NON_VERIFY_PHASES = {
    "approach", "slow_approach", "move_to",       # positioning
    "contact", "dwell", "drag", "release",        # touch/click execution
    "click",                                       # absolute click
    "type_text", "key_combo", "scroll_wheel",     # keyboard / scroll execution
}


# ── the worker ───────────────────────────────────────────────────────

class VerifierWorker:
    def tick(self, state: "RuntimeState") -> None:
        chunk = state.active_action_chunk
        if chunk is None:
            return
        if not chunk.expectation_ids:
            return

        # Only verify outcomes during the verify phase.
        current = state.gesture_state.current_phase
        if current in _NON_VERIFY_PHASES:
            return

        # ── score each branch ──
        branch_scores = {}
        for bid in list(state.live_branch_ids):
            b = state.branches.get(bid)
            if not b:
                continue
            scores = []
            for eid in b.expectation_ids:
                exp = state.expectations[eid]
                scores.append(score_expectation(exp, state))
            branch_scores[bid] = sum(scores) / max(len(scores), 1)

        if not branch_scores:
            return

        # ── model refinement ──
        x = torch.tensor([
            state.pointer_uncertainty,
            state.branch_entropy,
            state.fragile_action_phase,
            state.pending_timeout_pressure,
            1.0 if state.latest_delta is None else state.latest_delta.global_change_score,
        ] + [0.0] * 59, dtype=torch.float32)
        logits = state.models.verify_logits(x)
        probs = F.softmax(logits, dim=-1)
        model_bias = {
            "success": float(probs[0].item()),
            "partial": float(probs[1].item()),
            "failure": float(probs[2].item()),
            "ambiguous": float(probs[3].item()),
            "delayed": float(probs[4].item()),
        }

        # ── update branch posteriors ──
        Z = sum(max(v, 1e-4) for v in branch_scores.values())
        for bid, s in branch_scores.items():
            state.branches[bid].posterior = max(s, 1e-4) / Z

        best_bid, best = max(branch_scores.items(), key=lambda kv: kv[1])

        # ── multi-scale verification state ──
        vs = compute_verification_state(state, best)
        state.verification_state = vs

        # ── determine failure scale ──
        # Find the LOWEST scale that failed — that's where the problem is.
        # This lets recovery jump to the right level.
        scale_scores = {
            0: vs.micro_ok,
            1: vs.servo_ok,
            2: vs.gesture_ok,
            3: vs.interaction_ok,
            4: vs.subtask_ok,
            5: vs.task_ok,
        }

        if best > 0.78:
            label = "success"
        elif best > 0.58:
            label = "partial"
        elif best > 0.35:
            label = "ambiguous"
        else:
            label = "failure"

        # timeout + model bias → delayed
        elapsed_ms = (state.now_ns - chunk.started_ns) / 1e6
        if label in {"failure", "ambiguous"} and model_bias["delayed"] > 0.35 and elapsed_ms < chunk.timeout_ms:
            label = "delayed"

        # Identify which scale failed most
        scale_failures = {lvl: max(0.0, 1.0 - s) for lvl, s in scale_scores.items()}
        # Find the lowest scale with failure > 0.5
        failure_scale = None
        for lvl in range(6):
            if scale_failures.get(lvl, 0.0) > 0.5:
                failure_scale = lvl
                break

        verdict = VerifierVerdict(
            label=label,  # type: ignore[arg-type]
            scale_failures=scale_failures,
            branch_posteriors={bid: state.branches[bid].posterior for bid in branch_scores},
            notes={
                "best_branch": best_bid,
                "best_score": best,
                "model_bias": model_bias,
                "failure_scale": failure_scale,
                "verification_state": {
                    "micro_ok": vs.micro_ok,
                    "servo_ok": vs.servo_ok,
                    "gesture_ok": vs.gesture_ok,
                    "interaction_ok": vs.interaction_ok,
                    "subtask_ok": vs.subtask_ok,
                    "task_ok": vs.task_ok,
                },
            },
        )

        ev = state.event_journal.make_event(
            "verifier_judgment",
            state.session_id,
            state.episode_id,
            {"verdict": verdict},
        )
        state.event_journal.append(ev)

        if label in {"success", "partial"}:
            state.branches[best_bid].status = "supported"
            state.active_action_chunk = None
            state.gesture_state.active_chunk_id = None
            state.gesture_state.current_phase = None
            state.failure_density = max(0.0, state.failure_density - 0.15)
        elif label == "failure":
            state.branches[best_bid].status = "rejected"
            state.failure_density = min(1.0, state.failure_density + 0.25)
            # Annotate recovery with the failure scale for scale-aware recovery
            reason = "verification_failure"
            if failure_scale is not None:
                reason = f"verification_failure_L{failure_scale}"
            state.recovery_reasons.append(reason)
        elif label == "ambiguous":
            state.failure_density = min(1.0, state.failure_density + 0.05)
        else:
            pass
