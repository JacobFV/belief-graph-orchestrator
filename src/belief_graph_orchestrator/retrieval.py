
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .schemas import ComplexState, Workspace, WorkspaceNode
from .utils import bbox_center, dedupe_preserve_order, l2, topk


def _pad(vec: list[float] | None, dim: int) -> list[float]:
    if vec is None:
        return [0.0] * dim
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def _text_overlap_score(task_text: str, node_text: str) -> float:
    task_toks = set(t.lower() for t in task_text.replace("/", " ").replace("_", " ").split() if t.strip())
    node_toks = set(t.lower() for t in node_text.replace("/", " ").replace("_", " ").split() if t.strip())
    if not task_toks or not node_toks:
        return 0.0
    return len(task_toks & node_toks) / max(len(task_toks), 1)


def _scale_affinity(node_scale: int, query_scale: int) -> float:
    """Nodes at or near the query scale get a bonus; distant scales are penalized."""
    diff = abs(node_scale - query_scale)
    if diff == 0:
        return 0.5
    if diff == 1:
        return 0.2
    return 0.0


def score_frontier_nodes(state, complex_state: ComplexState, scale_level: int) -> list[tuple[int, float]]:
    scores = []
    frontier = set(state.interaction_state.candidate_target_ids) | set(state.subtask_state.target_region_ids)
    for nid in state.residency.resident_ids:
        node = state.graph.node(nid)
        s = 0.0
        if nid in frontier:
            s += 1.0
        s += _scale_affinity(node.scale, scale_level)
        s += 0.2 * node.confidence
        # At high scales (L4-L5), boost screen_region / route / goal nodes
        if scale_level >= 4 and node.kind in {"screen_region", "route_hypothesis", "goal_state", "task_subgoal"}:
            s += 0.4
        # At low scales (L0-L1), boost pointer / gesture nodes
        if scale_level <= 1 and node.kind in {"pointer_posterior", "micro_residual", "gesture_phase"}:
            s += 0.4
        scores.append((nid, s))
    return scores


def score_spatial_nodes(state, complex_state: ComplexState, scale_level: int) -> list[tuple[int, float]]:
    scores = []
    px, py = state.pointer.x_hat, state.pointer.y_hat
    # Scale determines spatial radius: low scale → tight, high scale → broad
    radius = 200.0 if scale_level <= 2 else 500.0 if scale_level <= 4 else 1000.0
    for nid in state.residency.resident_ids:
        node = state.graph.node(nid)
        bb = node.state.get("bbox")
        if bb is None:
            continue
        cx, cy = bbox_center(bb)
        d = l2((px, py), (cx, cy))
        s = max(0.0, 1.0 - d / radius)
        s += _scale_affinity(node.scale, scale_level)
        scores.append((nid, s))
    return scores


def score_causal_nodes(state, complex_state: ComplexState, scale_level: int) -> list[tuple[int, float]]:
    scores = []
    recent_failure_nodes = state.failure_anchor_node_ids[-8:]
    seeds = set(recent_failure_nodes)
    for nid in recent_failure_nodes:
        seeds.update(state.graph.reverse_neighbors(nid))
        seeds.update(state.graph.neighbors(nid))
    for nid in seeds:
        if nid in state.graph.nodes and nid in state.residency.resident_ids:
            node = state.graph.node(nid)
            s = 0.7 + 0.1 * node.confidence
            if node.scale == scale_level:
                s += 0.2
            scores.append((nid, s))
    return scores


def score_branch_nodes(state, complex_state: ComplexState, scale_level: int) -> list[tuple[int, float]]:
    scores = []
    for bid in state.live_branch_ids:
        b = state.branches.get(bid)
        if not b:
            continue
        for nid in b.node_ids:
            if nid in state.residency.resident_ids and nid in state.graph.nodes:
                node = state.graph.node(nid)
                s = b.posterior
                if node.scale == scale_level:
                    s += 0.15
                scores.append((nid, s))
    return scores


def score_semantic_nodes(state, complex_state: ComplexState, scale_level: int) -> list[tuple[int, float]]:
    scores = []
    task_text = state.task_state.active_goal or ""
    task_vec = state.task_embedding
    for nid in state.residency.resident_ids:
        node = state.graph.node(nid)
        text = node.state.get("text", "") or ""
        s = _text_overlap_score(task_text, text)
        if node.z_obj is not None:
            node_vec = torch.tensor(_pad(node.z_obj, 128), dtype=torch.float32).unsqueeze(0)
            ctx_vec = torch.tensor(task_vec if task_vec else [0.0] * 128, dtype=torch.float32)
            extra = torch.tensor([[node.confidence, node.scale, float(scale_level), node.state.get("actionable_prob", 0.0), 0, 0, 0, 0]], dtype=torch.float32)
            model_s = float(state.models.score_nodes(node_vec, ctx_vec, extra)[0].item())
            s += 0.05 * model_s
        if node.scale == scale_level:
            s += 0.2
        s += 0.2 * node.state.get("actionable_prob", 0.0)
        scores.append((nid, s))
    return scores


def score_historical_nodes(state, complex_state: ComplexState, scale_level: int) -> list[tuple[int, float]]:
    scores = []
    for nid in state.residency.resident_ids:
        node = state.graph.node(nid)
        if node.kind not in {"historic_anchor", "analogy_anchor", "failure_pattern", "episode_summary"}:
            continue
        s = node.confidence + 0.1 * node.state.get("utility_score", 0.0)
        scores.append((nid, s))
    return scores


def score_analogical_nodes(state, complex_state: ComplexState, scale_level: int) -> list[tuple[int, float]]:
    # For now identical to historical, but kept separate to preserve architecture.
    return score_historical_nodes(state, complex_state, scale_level)


def assemble_query_pool(state, complex_state: ComplexState, scale_level: int) -> list[int]:
    frontier = score_frontier_nodes(state, complex_state, scale_level)
    spatial = score_spatial_nodes(state, complex_state, scale_level)
    causal = score_causal_nodes(state, complex_state, scale_level)
    branch = score_branch_nodes(state, complex_state, scale_level)
    semantic = score_semantic_nodes(state, complex_state, scale_level)
    historical = score_historical_nodes(state, complex_state, scale_level)
    analogical = score_analogical_nodes(state, complex_state, scale_level)

    selected = []
    selected += topk(frontier, complex_state.budget.frontier_k)
    selected += topk(spatial, complex_state.budget.spatial_k)
    selected += topk(causal, complex_state.budget.causal_k)
    selected += topk(branch, complex_state.budget.branch_k)
    selected += topk(semantic, complex_state.budget.semantic_k)
    selected += topk(historical, complex_state.budget.historical_k)
    selected += topk(analogical, complex_state.budget.analogical_k)
    selected = dedupe_preserve_order(selected)

    for nid in selected:
        state.residency.touch(nid, state.now_ns)
    return selected


def pack_workspace(state, node_ids: list[int]) -> Workspace:
    nodes = []
    for nid in node_ids:
        n = state.graph.node(nid)
        nodes.append(
            WorkspaceNode(
                node_id=nid,
                scale=n.scale,
                kind=n.kind,
                bbox=n.state.get("bbox"),
                text=n.state.get("text"),
                confidence=n.confidence,
                state=n.state,
                z_obj=n.z_obj,
                z_dyn=n.z_dyn,
                z_belief=n.z_belief,
            )
        )

    return Workspace(
        task_state=state.task_state,
        subtask_state=state.subtask_state,
        interaction_state=state.interaction_state,
        gesture_state=state.gesture_state,
        servo_state=state.servo_state,
        branch_summary={
            "num_live": len(state.live_branch_ids),
            "entropy": state.branch_entropy,
        },
        nodes=nodes,
    )


def workspace_to_tokens(workspace: Workspace, dim: int = 192) -> torch.Tensor:
    tokens = []

    def tovec(xs, d):
        xs = list(xs)
        if len(xs) >= d:
            return xs[:d]
        return xs + [0.0] * (d - len(xs))

    # state tokens
    tokens.append(torch.tensor(tovec(workspace.task_state.z, dim), dtype=torch.float32))
    tokens.append(torch.tensor(tovec(workspace.subtask_state.z, dim), dtype=torch.float32))
    tokens.append(torch.tensor(tovec(workspace.interaction_state.z, dim), dtype=torch.float32))
    tokens.append(torch.tensor(tovec(workspace.gesture_state.z, dim), dtype=torch.float32))
    tokens.append(torch.tensor(tovec(workspace.servo_state.z, dim), dtype=torch.float32))

    for n in workspace.nodes:
        z_obj = _pad(n.z_obj, 128)
        z_dyn = _pad(n.z_dyn, 32)
        z_bel = _pad(n.z_belief, 16)
        head = [float(n.scale), float(n.confidence), float(n.state.get("actionable_prob", 0.0)), float(n.state.get("selected", False))]
        vec = z_obj + z_dyn + z_bel + head
        tokens.append(torch.tensor(tovec(vec, dim), dtype=torch.float32))

    if not tokens:
        return torch.zeros(1, 1, dim)
    return torch.stack(tokens, dim=0).unsqueeze(0)
