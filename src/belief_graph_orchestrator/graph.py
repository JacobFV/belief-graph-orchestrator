
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Optional

from .schemas import BeliefEdge, BeliefNode, EdgeKind, NodeKind
from .utils import now_ns


class BeliefGraph:
    def __init__(self) -> None:
        self.nodes: dict[int, BeliefNode] = {}
        self.edges: list[BeliefEdge] = []
        self.out_edges: dict[int, list[BeliefEdge]] = defaultdict(list)
        self.in_edges: dict[int, list[BeliefEdge]] = defaultdict(list)
        self._next_node_id = 1

    def next_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def add_node(self, node: BeliefNode) -> BeliefNode:
        self.nodes[node.id] = node
        self._next_node_id = max(self._next_node_id, node.id + 1)
        return node

    def create_node(
        self,
        scale: int,
        kind: NodeKind,
        confidence: float,
        state: dict,
        support_event_ids: Optional[list[int]] = None,
        support_node_ids: Optional[list[int]] = None,
        contradiction_event_ids: Optional[list[int]] = None,
        z_obj: Optional[list[float]] = None,
        z_dyn: Optional[list[float]] = None,
        z_belief: Optional[list[float]] = None,
        z_value: Optional[list[float]] = None,
    ) -> BeliefNode:
        node = BeliefNode(
            id=self.next_node_id(),
            scale=scale,
            version=1,
            kind=kind,
            status="active",
            support_event_ids=support_event_ids or [],
            support_node_ids=support_node_ids or [],
            contradiction_event_ids=contradiction_event_ids or [],
            t_start_ns=now_ns(),
            t_end_ns=None,
            confidence=confidence,
            state=state,
            z_obj=z_obj,
            z_dyn=z_dyn,
            z_belief=z_belief,
            z_value=z_value,
        )
        return self.add_node(node)

    def revise_node(self, node_id: int, **updates) -> BeliefNode:
        old = self.nodes[node_id]
        revised = replace(old, **updates, version=old.version + 1)
        if old.status == "active":
            old.status = "superseded"
            old.t_end_ns = now_ns()
        self.nodes[revised.id] = revised
        return revised

    def add_edge(self, edge: BeliefEdge) -> BeliefEdge:
        self.edges.append(edge)
        self.out_edges[edge.src].append(edge)
        self.in_edges[edge.dst].append(edge)
        return edge

    def connect(self, src: int, dst: int, kind: EdgeKind, confidence: float = 1.0) -> BeliefEdge:
        return self.add_edge(
            BeliefEdge(
                src=src,
                dst=dst,
                kind=kind,
                confidence=confidence,
                t_start_ns=now_ns(),
                t_end_ns=None,
            )
        )

    def node(self, node_id: int) -> BeliefNode:
        return self.nodes[node_id]

    def neighbors(self, node_id: int, kinds: Optional[set[EdgeKind]] = None) -> list[int]:
        edges = self.out_edges.get(node_id, [])
        if kinds is not None:
            edges = [e for e in edges if e.kind in kinds]
        return [e.dst for e in edges]

    def reverse_neighbors(self, node_id: int, kinds: Optional[set[EdgeKind]] = None) -> list[int]:
        edges = self.in_edges.get(node_id, [])
        if kinds is not None:
            edges = [e for e in edges if e.kind in kinds]
        return [e.src for e in edges]

    def active_nodes(self) -> list[int]:
        return [nid for nid, n in self.nodes.items() if n.status == "active"]

    def active_nodes_by_scale(self, scale: int) -> list[int]:
        return [nid for nid, n in self.nodes.items() if n.status == "active" and n.scale == scale]

    def nodes_by_kind(self, kind: NodeKind) -> list[int]:
        return [nid for nid, n in self.nodes.items() if n.kind == kind and n.status == "active"]

    def active_affordance_like(self) -> list[int]:
        out = []
        for nid, n in self.nodes.items():
            if n.status != "active":
                continue
            if n.kind in {"affordance", "candidate_target", "text_span", "container"}:
                out.append(nid)
        return out
