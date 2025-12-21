
from __future__ import annotations

from .schemas import ResidencyRecord
from .utils import cosine


class ResidencyManager:
    """
    Manages which graph nodes are "hot" (resident in fast memory).

    Uses factorized saliency scoring from the spec:
      s_presence  — how structurally real / stable this node is
      s_relevance — how relevant to the current task
      s_urgency   — how urgently it needs updating now

    The hot_score is a weighted combination. Nodes are promoted/demoted
    by hot_score, never semantically destroyed.
    """

    def __init__(self, max_hot_nodes: int = 2048) -> None:
        self.records: dict[int, ResidencyRecord] = {}
        self.max_hot_nodes = max_hot_nodes
        self.resident_ids: set[int] = set()

    def ensure(self, node_id: int, now_ns: int) -> None:
        if node_id not in self.records:
            self.records[node_id] = ResidencyRecord(
                node_id=node_id,
                hot_score=0.0,
                last_access_ns=now_ns,
                recent_access_count=1,
                frontier_score=0.0,
                branch_score=0.0,
                motor_score=0.0,
                historic_score=0.0,
                anomaly_score=0.0,
            )
        self.touch(node_id, now_ns)
        self.resident_ids.add(node_id)

    def touch(self, node_id: int, now_ns: int) -> None:
        rec = self.records.setdefault(
            node_id,
            ResidencyRecord(
                node_id=node_id,
                hot_score=0.0,
                last_access_ns=now_ns,
                recent_access_count=0,
                frontier_score=0.0,
                branch_score=0.0,
                motor_score=0.0,
                historic_score=0.0,
                anomaly_score=0.0,
            ),
        )
        rec.last_access_ns = now_ns
        rec.recent_access_count += 1
        self.resident_ids.add(node_id)

    def update_scores(self, graph, state) -> None:
        for nid in list(self.records):
            rec = self.records[nid]
            node = graph.nodes.get(nid)
            if node is None or node.status != "active":
                self.resident_ids.discard(nid)
                continue

            # ── factorized saliency ──
            s_presence = self._score_presence(node, state, rec)
            s_relevance = self._score_relevance(node, state)
            s_urgency = self._score_urgency(node, state, rec)

            rec.frontier_score = s_relevance
            rec.branch_score = self._branch_score(node, state)
            rec.motor_score = self._motor_score(node, state)
            rec.historic_score = self._historic_score(node, state)
            rec.anomaly_score = s_urgency

            # Composite hot score from factorized saliency
            rec.hot_score = (
                1.0 * s_presence +
                1.5 * s_relevance +
                1.2 * s_urgency +
                1.0 * rec.branch_score +
                1.4 * rec.motor_score +
                0.8 * rec.historic_score
            )

        self._trim()

    # ── factorized saliency (from spec) ──────────────────────────────

    def _score_presence(self, node, state, rec) -> float:
        """
        s_presence: how structurally real this node is.
        High for nodes with strong evidence, stability over frames,
        consistent containment, confident role.
        """
        score = node.confidence * 0.5
        # Stability: accessed multiple times → more real
        score += min(rec.recent_access_count, 10) * 0.05
        # Has text → more identifiable
        if node.state.get("text"):
            score += 0.2
        # Has clear role → more real
        role_probs = node.state.get("role_probs", {})
        max_role = max(role_probs.values()) if role_probs else 0.0
        score += 0.2 * max_role
        return min(1.0, score)

    def _score_relevance(self, node, state) -> float:
        """
        s_relevance: how relevant to the current task.
        Uses embedding similarity with task + frontier membership.
        """
        score = 0.0
        # Frontier membership
        if node.id in state.interaction_state.candidate_target_ids:
            score += 0.6
        if node.id in state.subtask_state.target_region_ids:
            score += 0.4
        # Embedding similarity to task
        if node.z_obj and state.task_embedding:
            sim = cosine(node.z_obj[:128], state.task_embedding[:128])
            score += 0.4 * max(0.0, sim)
        return min(1.0, score)

    def _score_urgency(self, node, state, rec) -> float:
        """
        s_urgency: how urgently this node needs updating.
        High for recently changed, newly appeared, uncertain,
        or anomalous nodes.
        """
        score = 0.0
        # Recency of last access (recently accessed = urgent)
        if state.now_ns > 0 and rec.last_access_ns > 0:
            age_ms = (state.now_ns - rec.last_access_ns) / 1e6
            if age_ms < 500:
                score += 0.5  # very recent
            elif age_ms < 2000:
                score += 0.2
        # Provenance: counterfactual/imagined nodes are more urgent (need verification)
        prov = node.state.get("provenance", "observed")
        if prov in {"counterfactual", "imagined"}:
            score += 0.4
        elif prov == "tracked":
            score += 0.2
        # Error/warning text
        text = (node.state.get("text") or "").lower()
        if "error" in text or "warning" in text or "fail" in text:
            score += 0.5
        return min(1.0, score)

    # ── component scores ─────────────────────────────────────────────

    def _branch_score(self, node, state) -> float:
        score = 0.0
        for bid in state.live_branch_ids:
            b = state.branches.get(bid)
            if b and node.id in b.node_ids:
                score += b.posterior
        return min(score, 1.0)

    def _motor_score(self, node, state) -> float:
        bb = node.state.get("bbox")
        if bb is None:
            return 0.0
        px, py = state.pointer.x_hat, state.pointer.y_hat
        x1, y1, x2, y2 = bb
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        return max(0.0, 1.0 - dist / 300.0)

    def _historic_score(self, node, state) -> float:
        if node.kind in {"historic_anchor", "analogy_anchor", "failure_pattern", "episode_summary"}:
            return node.confidence
        return 0.0

    def _trim(self) -> None:
        if len(self.resident_ids) <= self.max_hot_nodes:
            return
        scored = sorted(
            [(nid, self.records[nid].hot_score) for nid in self.resident_ids],
            key=lambda x: x[1],
            reverse=True,
        )
        keep = {nid for nid, _ in scored[: self.max_hot_nodes]}
        self.resident_ids = keep
