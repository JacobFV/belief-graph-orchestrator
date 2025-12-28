
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .graph import BeliefGraph
from .schemas import (
    BeliefNode,
    Branch,
    ComplexState,
    DeltaFeatures,
    LayoutHint,
    OCRSpan,
    PointerCandidate,
    RegionProposal,
)
from .utils import bbox_center, cosine, iou, now_ns


def role_probs_for_label(label: Optional[str]) -> dict[str, float]:
    label = (label or "unknown").lower()
    if label in {"button"}:
        return {"ActionableDiscrete": 0.9, "NavigationBack": 0.1}
    if label in {"toggle"}:
        return {"SelectionControl": 0.9, "ActionableDiscrete": 0.6}
    if label in {"text_field"}:
        return {"TextEntry": 0.95}
    if label in {"list_item"}:
        return {"CollectionItem": 0.9, "ActionableDiscrete": 0.4}
    if label in {"label"}:
        return {"DisplayOnlyText": 0.95}
    if label in {"nav_bar", "tab_bar", "sheet", "modal", "list", "keyboard", "toolbar", "menubar", "sidebar", "dialog", "statusbar"}:
        return {"HierarchicalContainer": 0.9}
    return {"ActionableDiscrete": 0.2}


def role_compatibility(role_probs: dict[str, float], label_hint: Optional[str]) -> float:
    if not role_probs:
        return 0.0
    hint = (label_hint or "").lower()
    if hint == "button":
        return role_probs.get("ActionableDiscrete", 0.0)
    if hint == "toggle":
        return role_probs.get("SelectionControl", 0.0)
    if hint == "text_field":
        return role_probs.get("TextEntry", 0.0)
    if hint == "label":
        return role_probs.get("DisplayOnlyText", 0.0)
    if hint in {"nav_bar", "tab_bar", "sheet", "modal", "list", "keyboard", "toolbar", "menubar", "sidebar", "dialog", "statusbar"}:
        return role_probs.get("HierarchicalContainer", 0.0)
    return 0.1


class BeliefWorker:
    def __init__(self) -> None:
        self.recent_failures: list[tuple[int, str]] = []

    def step(self, state: "RuntimeState", new_events: list) -> None:
        buckets = bucket_events(new_events)

        self.update_entities_from_proposals(state, buckets.get("regions", []))
        self.attach_ocr_spans(state, buckets.get("ocr", []))
        self.update_layout_and_containers(state, buckets.get("layout", []))
        self.update_pointer_hypotheses(state, buckets.get("pointer_candidates", []))
        self.update_delta_and_outcomes(state, buckets.get("delta", []))
        self.update_live_branches(state)

    def update_entities_from_proposals(self, state: "RuntimeState", region_events: list) -> None:
        graph = state.graph
        active_entities = [
            nid for nid in graph.active_affordance_like()
            if graph.node(nid).kind in {"affordance", "candidate_target", "container", "text_span"}
        ]

        for ev in region_events:
            for prop in ev.payload.get("proposals", []):
                if isinstance(prop, dict):
                    continue
                nid = associate_proposal_to_entities(prop, graph, active_entities)
                if nid is None:
                    self.create_new_entity(state, prop, ev.id)
                else:
                    self.revise_entity(state, nid, prop, ev.id)

    def _fuse_canonical_embedding(
        self,
        state: "RuntimeState",
        patch_emb: list[float],
        text: str | None,
        role_probs: dict[str, float],
        bbox: tuple,
        scale: int,
        prop_score: float,
    ) -> list[float]:
        """
        Multi-trunk canonical encoder (from the spec):
          - Visual trunk: patch embedding from TinyVisionEncoder
          - Text trunk: text embedding from TextEncoder
          - Structural trunk: scale, relative position, area, aspect ratio
          - Role trunk: actionability, text-entry-ness, container-ness

        Fuses into a single z_obj vector by concatenation + learned projection
        through the NodeScorer's internal representation.
        """
        import torch

        # Visual trunk: already have patch_emb (128-d)
        vis = patch_emb[:128] if len(patch_emb) >= 128 else patch_emb + [0.0] * (128 - len(patch_emb))

        # Text trunk: embed the node's text (128-d → take first 64 for budget)
        if text and len(text) > 1:
            text_emb = state.models.encode_text([text])[0].cpu().tolist()[:64]
        else:
            text_emb = [0.0] * 64

        # Structural trunk: geometry + scale features
        x1, y1, x2, y2 = bbox
        body_w = getattr(state.body, "width", 1280) or 1280
        body_h = getattr(state.body, "height", 800) or 800
        bw, bh = x2 - x1, y2 - y1
        structural = [
            float(scale) / 5.0,
            x1 / body_w, y1 / body_h,              # relative position
            bw / body_w, bh / body_h,              # relative size
            bw / max(bh, 1),                        # aspect ratio
            (bw * bh) / (body_w * body_h),          # area fraction
        ]

        # Role trunk: affordance priors as a dense vector
        role_vec = [
            role_probs.get("ActionableDiscrete", 0.0),
            role_probs.get("TextEntry", 0.0),
            role_probs.get("SelectionControl", 0.0),
            role_probs.get("HierarchicalContainer", 0.0),
            role_probs.get("DisplayOnlyText", 0.0),
            role_probs.get("CollectionItem", 0.0),
            prop_score,
        ]

        # Fuse: visual (128) + text (64) + structural (7) + role (7) = 206
        # Project down to 128 via simple learned linear (or just truncate/hash for now)
        # For now: weight-averaged concatenation keeping 128-d
        fused = []
        for i in range(128):
            v = vis[i] if i < len(vis) else 0.0
            t = text_emb[i % 64] * 0.3 if text_emb else 0.0
            s = structural[i % 7] * 0.15 if i < 14 else 0.0
            r = role_vec[i % 7] * 0.1 if i < 14 else 0.0
            fused.append(v * 0.6 + t + s + r)
        return fused

    def create_new_entity(self, state: "RuntimeState", prop: RegionProposal, support_event_id: int) -> int:
        label = (prop.label_hint or "").lower()
        kind = "container" if label in {"nav_bar", "tab_bar", "sheet", "modal", "list", "keyboard", "toolbar", "menubar", "sidebar", "dialog", "statusbar"} else "affordance"
        scale = 4 if kind == "container" else 3
        role_probs = role_probs_for_label(prop.label_hint)
        text = prop.metadata.get("text") if prop.metadata else None

        # Multi-trunk canonical embedding
        z_obj = self._fuse_canonical_embedding(
            state, prop.patch_embedding, text, role_probs, prop.bbox, scale, prop.score,
        )

        node = state.graph.create_node(
            scale=scale,
            kind=kind,  # type: ignore[arg-type]
            confidence=prop.score,
            state={
                "bbox": prop.bbox,
                "role_probs": role_probs,
                "label_hint": prop.label_hint,
                "text": text,
                "actionable_prob": role_probs.get("ActionableDiscrete", 0.0),
                "selected": bool(prop.metadata.get("selected", False)) if prop.metadata else False,
                "provenance": "observed",
            },
            support_event_ids=[support_event_id],
            z_obj=z_obj,
            z_dyn=[prop.score, 0.0, 0.0, 0.0],
            z_belief=[prop.score, 0.0, 0.0, 0.0],
            z_value=[role_probs.get("ActionableDiscrete", 0.0), role_probs.get("TextEntry", 0.0)],
        )
        state.residency.ensure(node.id, state.now_ns)
        return node.id

    def revise_entity(self, state: "RuntimeState", node_id: int, prop: RegionProposal, support_event_id: int) -> None:
        node = state.graph.node(node_id)
        old_conf = node.confidence
        old_role = dict(node.state.get("role_probs", {}))
        node.state["bbox"] = prop.bbox
        node.state["label_hint"] = prop.label_hint or node.state.get("label_hint")
        if prop.metadata.get("text"):
            node.state["text"] = prop.metadata["text"]
        node.state["selected"] = bool(prop.metadata.get("selected", node.state.get("selected", False)))
        node.state["role_probs"] = merge_role_probs(node.state.get("role_probs", {}), role_probs_for_label(prop.label_hint))
        node.state["actionable_prob"] = max(node.state["role_probs"].get("ActionableDiscrete", 0.0), node.state["role_probs"].get("SelectionControl", 0.0))
        node.confidence = min(1.0, 0.7 * node.confidence + 0.3 * prop.score)
        node.support_event_ids.append(support_event_id)
        # Re-fuse canonical embedding with updated text/role/geometry
        node.z_obj = self._fuse_canonical_embedding(
            state, prop.patch_embedding,
            node.state.get("text"), node.state.get("role_probs", {}),
            prop.bbox, node.scale, prop.score,
        )
        node.state["provenance"] = "observed"  # refreshed by direct observation
        state.residency.touch(node_id, state.now_ns)

        # Emit belief_revision event for audit trail
        ev = state.event_journal.make_event(
            "belief_revision",
            state.session_id,
            state.episode_id,
            {
                "node_id": node_id,
                "revision": "entity_update",
                "old_confidence": old_conf,
                "new_confidence": node.confidence,
                "role_changed": old_role != node.state.get("role_probs", {}),
            },
            parent_ids=[support_event_id],
        )
        state.event_journal.append(ev)

    def attach_ocr_spans(self, state: "RuntimeState", ocr_events: list) -> None:
        graph = state.graph
        for ev in ocr_events:
            for span in ev.payload.get("spans", []):
                if isinstance(span, dict):
                    continue
                best = None
                best_iou = 0.0
                for nid in graph.active_affordance_like():
                    n = graph.node(nid)
                    bb = n.state.get("bbox")
                    ov = iou(bb, span.bbox)
                    if ov > best_iou:
                        best_iou = ov
                        best = nid
                if best is not None and best_iou > 0.15:
                    n = graph.node(best)
                    n.state["text"] = span.text
                    n.state["ocr_conf"] = span.confidence
                    n.support_event_ids.append(ev.id)
                    state.residency.touch(best, state.now_ns)
                else:
                    node = graph.create_node(
                        scale=3,
                        kind="text_span",
                        confidence=span.confidence,
                        state={"bbox": span.bbox, "text": span.text, "role_probs": {"DisplayOnlyText": 1.0}},
                        support_event_ids=[ev.id],
                    )
                    state.residency.ensure(node.id, state.now_ns)

    def update_layout_and_containers(self, state: "RuntimeState", layout_events: list) -> None:
        graph = state.graph
        for ev in layout_events:
            for hint in ev.payload.get("layout_hints", []):
                if isinstance(hint, dict):
                    continue
                best = None
                best_i = 0.0
                for nid in graph.nodes_by_kind("container"):
                    bb = graph.node(nid).state.get("bbox")
                    ov = iou(bb, hint.bbox)
                    if ov > best_i:
                        best_i = ov
                        best = nid
                if best is not None and best_i > 0.4:
                    node = graph.node(best)
                    node.state["layout_kind"] = hint.kind
                    node.confidence = max(node.confidence, hint.confidence)
                    node.support_event_ids.append(ev.id)
                    state.residency.touch(best, state.now_ns)
                else:
                    node = graph.create_node(
                        scale=4,
                        kind="container",
                        confidence=hint.confidence,
                        state={"bbox": hint.bbox, "layout_kind": hint.kind, "role_probs": {"HierarchicalContainer": 1.0}},
                        support_event_ids=[ev.id],
                    )
                    state.residency.ensure(node.id, state.now_ns)

    def update_pointer_hypotheses(self, state: "RuntimeState", ptr_events: list) -> None:
        best: Optional[PointerCandidate] = None
        for ev in ptr_events:
            for cand in ev.payload.get("pointer_candidates", []):
                if isinstance(cand, dict):
                    continue
                if best is None or cand.confidence > best.confidence:
                    best = cand
        state.latest_pointer_candidate = best

    def update_delta_and_outcomes(self, state: "RuntimeState", delta_events: list) -> None:
        if not delta_events:
            return
        latest = delta_events[-1].payload.get("delta")
        if latest is None or isinstance(latest, dict):
            return
        state.latest_delta = latest

    def update_live_branches(self, state: "RuntimeState") -> None:
        # Keep stale branches from lingering forever.
        if state.active_action_chunk is None:
            return
        cutoff_ns = state.now_ns - 3_000_000_000
        for bid in state.live_branch_ids:
            b = state.branches[bid]
            action_event = state.event_journal.get(b.root_action_event_id)
            if action_event.t_capture_ns < cutoff_ns and b.status == "live":
                b.status = "stale"


def bucket_events(events: list) -> dict[str, list]:
    out = defaultdict(list)
    for e in events:
        out[e.type].append(e)
    return out


def merge_role_probs(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = max(out.get(k, 0.0), v)
    return out


def temporal_continuity(node: BeliefNode) -> float:
    age = max(1, now_ns() - node.t_start_ns)
    return min(1.0, 1e9 / age)


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
