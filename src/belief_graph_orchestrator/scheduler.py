
from __future__ import annotations

from .schemas import ScaleBand


class Scheduler:
    def _due(self, cx, now_ns: int) -> bool:
        if cx.base_tick_hz <= 0:
            return False
        period_ns = int(1e9 / cx.base_tick_hz)
        return now_ns - cx.last_tick_ns >= period_ns

    def compute_frontier_hazard(self, state: "RuntimeState") -> float:
        return min(
            1.0,
            0.45 * state.pointer_uncertainty +
            0.20 * state.branch_entropy +
            0.20 * state.fragile_action_phase +
            0.15 * state.pending_timeout_pressure,
        )

    def compute_value_of_retrospection(self, state: "RuntimeState") -> float:
        return (
            0.30 * state.failure_density +
            0.25 * state.branch_entropy +
            0.20 * state.ambiguity_score +
            0.20 * state.analogy_match_score -
            0.30 * state.fragile_action_phase -
            0.25 * state.pointer_uncertainty
        )

    def reanchor_if_stale(self, state: "RuntimeState", frontier_hazard: float) -> None:
        if frontier_hazard < 0.8:
            return
        if state.interaction_state.candidate_target_ids:
            anchors = state.interaction_state.candidate_target_ids[:4]
            for key in ("task", "recovery", "historical_reflection"):
                if key in state.complexes:
                    state.complexes[key].anchor_node_ids = anchors

    # ── dynamic scale zooming ────────────────────────────────────────
    #
    # The spec says: "complexes should zoom in/out based on uncertainty."
    # Instead of static scale bands, we adapt them each tick based on
    # pointer uncertainty, branch entropy, failure density, and hazard.

    def adapt_scale_bands(self, state: "RuntimeState", frontier_hazard: float, vor: float) -> None:
        """Dynamically shift complex scale bands based on current state."""

        # Pointer: zoom tighter when uncertainty is high
        if "pointer" in state.complexes:
            cx = state.complexes["pointer"]
            if state.pointer_uncertainty > 0.5:
                # Zoom into L0 — tighten control
                cx.active_scale_band = ScaleBand({0: 0.8, 1: 0.2}, (5, 80), 1, 1)
            else:
                cx.active_scale_band = ScaleBand({0: 0.7, 1: 0.3}, (5, 120), 1, 1)

        # Task: zoom out when branch entropy or failure density are high
        if "task" in state.complexes:
            cx = state.complexes["task"]
            if state.branch_entropy > 0.6 or state.failure_density > 0.5:
                # Zoom out to reconsider the route / plan
                cx.active_scale_band = ScaleBand({3: 0.1, 4: 0.35, 5: 0.55}, (1000, 60000), 10, 8)
            elif frontier_hazard < 0.3:
                # Stable — can focus on interaction-level
                cx.active_scale_band = ScaleBand({3: 0.35, 4: 0.45, 5: 0.20}, (500, 20000), 8, 6)
            else:
                cx.active_scale_band = ScaleBand({3: 0.2, 4: 0.45, 5: 0.35}, (500, 30000), 8, 6)

        # Verifier: span broader when there are active expectations across scales
        if "verifier" in state.complexes:
            cx = state.complexes["verifier"]
            if state.active_action_chunk is not None:
                # During active chunk, focus on L1-L3 (motor through interaction)
                cx.active_scale_band = ScaleBand({1: 0.3, 2: 0.4, 3: 0.3}, (40, 1000), 4, 4)
            else:
                cx.active_scale_band = ScaleBand({1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2, 5: 0.1}, (40, 1500), 4, 4)

        # Recovery: cross-scale jumper — adapt based on latest failure scale
        if "recovery" in state.complexes:
            cx = state.complexes["recovery"]
            # Check if we have scale-specific failure info
            failure_scale = None
            for reason in state.recovery_reasons:
                if reason.startswith("verification_failure_L"):
                    try:
                        failure_scale = int(reason.split("_L")[1])
                    except (ValueError, IndexError):
                        pass
            if failure_scale is not None:
                # Focus recovery around the failure scale ± 1
                lo = max(0, failure_scale - 1)
                hi = min(5, failure_scale + 2)
                probs = {lvl: 1.0 / (hi - lo) for lvl in range(lo, hi)}
                cx.active_scale_band = ScaleBand(probs, (100, 10000), 8, 8)
            else:
                cx.active_scale_band = ScaleBand({2: 0.2, 3: 0.3, 4: 0.3, 5: 0.2}, (100, 10000), 8, 8)

        # Historical reflection: zoom broader when VoR is high
        if "historical_reflection" in state.complexes:
            cx = state.complexes["historical_reflection"]
            if vor > 0.3:
                # Strong reason to reflect — broader scale access
                cx.active_scale_band = ScaleBand({3: 0.15, 4: 0.35, 5: 0.50}, (500, 60000), 14, 12)
            else:
                cx.active_scale_band = ScaleBand({3: 0.1, 4: 0.4, 5: 0.5}, (500, 60000), 12, 10)

    def step(self, state: "RuntimeState", now_ns: int) -> list[tuple[str, int]]:
        # ── dynamic scale adaptation ──
        frontier_hazard = self.compute_frontier_hazard(state)
        vor = self.compute_value_of_retrospection(state)
        self.adapt_scale_bands(state, frontier_hazard, vor)

        ready: list[tuple[str, int]] = []
        for cx in state.complexes.values():
            if self._due(cx, now_ns):
                for lvl, prob in cx.active_scale_band.level_probs.items():
                    if prob > 0.05:
                        ready.append((cx.id, int(lvl)))

        mandatory = []
        for cid, lvl in ready:
            if cid in {"pointer", "verifier"}:
                mandatory.append((cid, lvl))
            if cid == "action_exec" and state.active_action_chunk is not None:
                mandatory.append((cid, lvl))

        runnable = list(dict.fromkeys(mandatory))

        for cid, lvl in ready:
            if (cid, lvl) in runnable:
                continue
            if cid == "historical_reflection" and vor <= 0.0:
                continue
            if frontier_hazard > 0.75 and cid == "historical_reflection":
                continue
            if frontier_hazard > 0.7 and lvl >= 4 and cid not in {"task", "recovery"}:
                continue
            runnable.append((cid, lvl))

        self.reanchor_if_stale(state, frontier_hazard)
        return runnable


PRIORITY = {
    "pointer": 0,
    "action_exec": 1,
    "verifier": 2,
    "frontier_visual": 3,
    "recovery": 4,
    "task": 5,
    "historical_reflection": 6,
    "background": 7,
}


def order_runnable(runnable: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return sorted(runnable, key=lambda x: (PRIORITY.get(x[0], 99), x[1]))
