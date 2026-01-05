
from __future__ import annotations

def compute_vor(state: "RuntimeState") -> float:
    return (
        0.30 * state.failure_density +
        0.25 * state.branch_entropy +
        0.20 * state.ambiguity_score +
        0.20 * state.analogy_match_score -
        0.30 * state.fragile_action_phase -
        0.25 * state.pointer_uncertainty
    )


class HistoricalReflectionWorker:
    def tick(self, state: "RuntimeState", scale_level: int = 5) -> None:
        if not state.force_reflection and compute_vor(state) <= 0.0:
            return
        state.force_reflection = False

        anchors = []
        for nid in list(state.residency.resident_ids):
            node = state.graph.node(nid)
            if node.kind in {"historic_anchor", "analogy_anchor", "failure_pattern", "episode_summary"}:
                anchors.append(nid)
        if not anchors:
            return

        # current implementation is heuristic:
        # if repeated failure, increase back preference and reduce aggressive control.
        state.subtask_state.contradiction_score = min(1.0, state.subtask_state.contradiction_score + 0.15)
        state.pointer.dynamics["gain_x"] = 0.95 * state.pointer.dynamics.get("gain_x", 1.0)
        state.pointer.dynamics["gain_y"] = 0.95 * state.pointer.dynamics.get("gain_y", 1.0)
