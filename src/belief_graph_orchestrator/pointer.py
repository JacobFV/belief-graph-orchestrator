
from __future__ import annotations

from typing import Any, Optional

import torch

from .schemas import ActionChunk, PointerCandidate, PointerPosterior
from .utils import clip, inflate_cov, shrink_cov


# ── Kalman predict / correct (velocity-servo mode) ──────────────────

def predict_pointer(pp: PointerPosterior, last_cmd: dict[str, float], dt_s: float) -> PointerPosterior:
    gain_x = pp.dynamics.get("gain_x", 1.0)
    gain_y = pp.dynamics.get("gain_y", 1.0)
    x = pp.x_hat + gain_x * last_cmd.get("vx", 0.0) * dt_s * 20.0
    y = pp.y_hat + gain_y * last_cmd.get("vy", 0.0) * dt_s * 20.0
    vx = last_cmd.get("vx", 0.0)
    vy = last_cmd.get("vy", 0.0)
    cov = inflate_cov(pp.cov, q=0.5)
    return PointerPosterior(x, y, vx, vy, cov, pp.visible_conf * 0.98, pp.last_obs_event_id, pp.dynamics)


def correct_pointer(pp: PointerPosterior, cand: Optional[PointerCandidate]) -> PointerPosterior:
    if cand is None or cand.confidence < 0.3:
        return pp
    alpha = min(max(cand.confidence, 0.1), 0.9)
    x = alpha * cand.x + (1.0 - alpha) * pp.x_hat
    y = alpha * cand.y + (1.0 - alpha) * pp.y_hat
    cov = shrink_cov(pp.cov, factor=(0.4 + 0.5 * cand.confidence))
    return PointerPosterior(x, y, pp.vx_hat, pp.vy_hat, cov, cand.confidence, pp.last_obs_event_id, pp.dynamics)


# ── target sampling ──────────────────────────────────────────────────

def sample_target_xy(target_distribution: dict[str, Any], pp: PointerPosterior) -> tuple[float, float]:
    if "mean" in target_distribution:
        x, y = target_distribution["mean"]
        return float(x), float(y)
    if "bbox" in target_distribution:
        x1, y1, x2, y2 = target_distribution["bbox"]
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return pp.x_hat, pp.y_hat


# ── learned residual (velocity mode only) ────────────────────────────

def learned_residual(state: "RuntimeState", target_xy: tuple[float, float]) -> tuple[float, float]:
    tx, ty = target_xy
    pp = state.pointer
    x = torch.tensor([
        pp.x_hat, pp.y_hat, pp.vx_hat, pp.vy_hat,
        tx, ty,
        pp.visible_conf,
        pp.cov[0][0], pp.cov[1][1],
        1.0,
    ], dtype=torch.float32)
    rv = state.models.pointer_resid(x)
    return float(rv[0].item()), float(rv[1].item())


def compute_velocity(state: "RuntimeState", target_xy: tuple[float, float], profile: dict[str, Any]) -> tuple[float, float]:
    pp = state.pointer
    tx, ty = target_xy
    ex = tx - pp.x_hat
    ey = ty - pp.y_hat
    d = (ex * ex + ey * ey) ** 0.5
    if d < profile.get("deadband_px", 3.0):
        return 0.0, 0.0

    gain = profile["far_gain"] if d > 40 else profile["near_gain"]
    if state.pointer.visible_conf < 0.2:
        gain *= 0.7

    vx = clip(gain * ex, -profile["max_vel"], profile["max_vel"])
    vy = clip(gain * ey, -profile["max_vel"], profile["max_vel"])

    rvx, rvy = learned_residual(state, target_xy)
    vx = clip(vx + rvx, -profile["max_vel"], profile["max_vel"])
    vy = clip(vy + rvy, -profile["max_vel"], profile["max_vel"])
    return vx, vy


# ── phase timing ─────────────────────────────────────────────────────

def current_phase(chunk: ActionChunk, now_ns: int) -> tuple[str, dict]:
    if not chunk.phases:
        return "idle", {}
    elapsed_ms = (now_ns - chunk.started_ns) / 1e6
    cursor = 0.0
    for ph in chunk.phases:
        dur = phase_duration_ms(ph)
        cursor += dur
        if elapsed_ms <= cursor:
            return ph.name, ph.params
    return chunk.phases[-1].name, chunk.phases[-1].params


def phase_duration_ms(phase) -> float:
    name = phase.name
    # ── velocity-servo phases (phone / hidden-cursor targets) ──
    if name in {"approach", "slow_approach"}:
        return 180.0
    if name == "contact":
        return 50.0
    if name == "dwell":
        return float(phase.params.get("ms", 60))
    if name == "drag":
        return 280.0
    if name == "release":
        return 60.0
    if name in {"verify", "verify_displacement"}:
        return float(phase.params.get("window_ms", 400))
    # ── absolute-mode phases (desktop) ──
    if name == "move_to":
        return 50.0
    if name == "click":
        return 30.0
    if name == "type_text":
        text = phase.params.get("text", "")
        return max(50.0, len(text) * 30.0)
    if name == "key_combo":
        return 50.0
    if name == "scroll_wheel":
        return 100.0
    return 50.0


# ── velocity-mode HID translation ───────────────────────────────────

def phase_to_hid_command(phase_name: str, phase_params: dict, vx: float, vy: float) -> dict[str, float | bool | int]:
    if phase_name in {"approach", "slow_approach"}:
        return {"vx": vx, "vy": vy, "contact": False, "button_mask": 1}
    if phase_name == "contact":
        return {"vx": 0.0, "vy": 0.0, "contact": True, "button_mask": 1}
    if phase_name == "dwell":
        return {"vx": 0.0, "vy": 0.0, "contact": True, "button_mask": 1}
    if phase_name == "drag":
        direction = phase_params.get("direction", "down")
        drag_vel = float(phase_params.get("amount", 0.5)) * 0.7
        if direction == "down":
            return {"vx": 0.0, "vy": drag_vel, "contact": True, "button_mask": 1}
        if direction == "up":
            return {"vx": 0.0, "vy": -drag_vel, "contact": True, "button_mask": 1}
        if direction == "left":
            return {"vx": -drag_vel, "vy": 0.0, "contact": True, "button_mask": 1}
        return {"vx": drag_vel, "vy": 0.0, "contact": True, "button_mask": 1}
    if phase_name == "release":
        return {"vx": 0.0, "vy": 0.0, "contact": False, "button_mask": 1}
    return {"vx": 0.0, "vy": 0.0, "contact": False, "button_mask": 1}


# ── the worker ───────────────────────────────────────────────────────

class PointerServoWorker:

    def __init__(self) -> None:
        self._last_phase_idx: int = -1  # track last executed phase for catch-up

    def tick(self, state: "RuntimeState", dt_s: float) -> None:
        body = state.body

        # ── direct-cursor mode (desktop/simulator) ──
        if body.has_direct_cursor:
            pos = body.get_cursor_position()
            if pos is not None:
                x, y = pos
                state.pointer = PointerPosterior(
                    x_hat=x, y_hat=y, vx_hat=0.0, vy_hat=0.0,
                    cov=[[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]],
                    visible_conf=1.0,
                    last_obs_event_id=state.pointer.last_obs_event_id,
                    dynamics=state.pointer.dynamics,
                )
                state.pointer_uncertainty = 0.0
        else:
            # ── velocity-servo mode (phone / hidden-cursor targets) ──
            pp = predict_pointer(state.pointer, state.last_hid_cmd, dt_s)
            pp = correct_pointer(pp, state.latest_pointer_candidate)
            state.pointer = pp
            state.pointer_uncertainty = min(1.0, 0.25 * (pp.cov[0][0] + pp.cov[1][1]) / 100.0)

        chunk = state.active_action_chunk
        if chunk is None:
            self._last_phase_idx = -1
            return

        phase_name, phase_params = current_phase(chunk, state.now_ns)
        state.gesture_state.current_phase = phase_name

        # ── dispatch to the right motor pathway ──
        if body.supports_absolute_move:
            self._execute_absolute_with_catchup(state, chunk, phase_name, phase_params)
        else:
            self._execute_velocity(state, chunk, phase_name, phase_params)

    # ── velocity pathway (phone / hidden-cursor targets) ─────────────

    def _execute_velocity(self, state: "RuntimeState", chunk: ActionChunk, phase_name: str, phase_params: dict) -> None:
        pp = state.pointer
        if phase_name in {"approach", "slow_approach"}:
            target_xy = sample_target_xy(chunk.target_distribution, pp)
            vx, vy = compute_velocity(state, target_xy, chunk.velocity_profile)
        else:
            vx, vy = 0.0, 0.0

        cmd = phase_to_hid_command(phase_name, phase_params, vx, vy)
        state.last_hid_cmd = {"vx": float(cmd["vx"]), "vy": float(cmd["vy"])}
        state.body.send_hid(
            vx=float(cmd["vx"]),
            vy=float(cmd["vy"]),
            contact=bool(cmd["contact"]),
            button_mask=int(cmd["button_mask"]),
        )
        self._emit_action_event(state, chunk, phase_name, cmd)

    # ── absolute pathway (desktop / direct-cursor targets) ───────────
    #
    # Absolute commands (move, click, key) are instantaneous.  If the tick
    # rate is slower than a phase duration, we might skip a side-effectful
    # phase entirely (e.g. a 30ms click phase between two ticks 100ms apart).
    # To prevent this, we track the last-executed phase index and catch up
    # on any skipped phases that have side effects.

    # Phases whose side effects MUST fire even if the tick skips past them.
    _SIDE_EFFECTFUL = {"click", "contact", "dwell", "type_text", "key_combo", "scroll_wheel"}

    def _execute_absolute_with_catchup(self, state: "RuntimeState", chunk: ActionChunk, phase_name: str, phase_params: dict) -> None:
        # Find the index of the current phase
        cur_idx = 0
        for i, ph in enumerate(chunk.phases):
            if ph.name == phase_name:
                cur_idx = i
                break

        # Execute any skipped side-effectful phases between last and current
        if self._last_phase_idx < cur_idx:
            for i in range(max(0, self._last_phase_idx + 1), cur_idx):
                ph = chunk.phases[i]
                if ph.name in self._SIDE_EFFECTFUL:
                    self._execute_absolute_phase(state, chunk, ph.name, ph.params)

        # Execute the current phase
        self._execute_absolute_phase(state, chunk, phase_name, phase_params)
        self._last_phase_idx = cur_idx

    def _execute_absolute_phase(self, state: "RuntimeState", chunk: ActionChunk, phase_name: str, phase_params: dict) -> None:
        body = state.body
        pp = state.pointer

        cmd: dict[str, Any] = {"phase": phase_name}

        if phase_name in {"move_to", "approach", "slow_approach"}:
            target_xy = sample_target_xy(chunk.target_distribution, pp)
            body.move_cursor_to(*target_xy)
            cmd.update({"type": "move_to", "x": target_xy[0], "y": target_xy[1]})

        elif phase_name == "click":
            target_xy = sample_target_xy(chunk.target_distribution, pp)
            button = phase_params.get("button", "left")
            body.click(*target_xy, button=button)
            cmd.update({"type": "click", "x": target_xy[0], "y": target_xy[1], "button": button})

        elif phase_name in {"contact", "dwell"}:
            target_xy = sample_target_xy(chunk.target_distribution, pp)
            body.click(*target_xy)
            cmd.update({"type": "click", "x": target_xy[0], "y": target_xy[1]})

        elif phase_name == "type_text":
            text = phase_params.get("text", "")
            if text:
                body.send_text(text)
            cmd.update({"type": "text", "text": text})

        elif phase_name == "key_combo":
            key = phase_params.get("key", "")
            modifiers = phase_params.get("modifiers", [])
            body.send_key(key, modifiers)
            cmd.update({"type": "key", "key": key, "modifiers": modifiers})

        elif phase_name == "scroll_wheel":
            direction = phase_params.get("direction", "down")
            delta = float(phase_params.get("delta", 3))
            dy = delta if direction == "down" else -delta
            dx = delta if direction == "right" else (-delta if direction == "left" else 0.0)
            body.send_hid(vx=dx, vy=dy, contact=False, button_mask=0)
            cmd.update({"type": "scroll", "direction": direction, "delta": delta})

        elif phase_name in {"release", "verify", "verify_displacement"}:
            cmd.update({"type": "noop", "phase": phase_name})

        else:
            cmd.update({"type": "noop", "phase": phase_name})

        state.last_hid_cmd = {"vx": 0.0, "vy": 0.0}
        self._emit_action_event(state, chunk, phase_name, cmd)

    # ── shared event emission ────────────────────────────────────────

    def _emit_action_event(self, state: "RuntimeState", chunk: ActionChunk, phase_name: str, cmd: dict) -> None:
        pp = state.pointer
        ev = state.event_journal.make_event(
            "action_issued",
            state.session_id,
            state.episode_id,
            {
                "chunk_id": chunk.id,
                "chunk_kind": chunk.kind,
                "phase": phase_name,
                "cmd": cmd,
                "pointer_estimate": {"x": pp.x_hat, "y": pp.y_hat},
                "target_node_ids": list(getattr(chunk, "target_node_ids", [])),
                "target_distribution": dict(getattr(chunk, "target_distribution", {})),
                "intent_confidence": float(getattr(chunk, "intent_confidence", 0.0)),
            },
        )
        state.event_journal.append(ev)
        if chunk.id not in state.chunk_root_event_ids:
            state.chunk_root_event_ids[chunk.id] = ev.id
            state.live_branch_ids = state.compiler.create_branches_for_action(ev.id, chunk, state)
