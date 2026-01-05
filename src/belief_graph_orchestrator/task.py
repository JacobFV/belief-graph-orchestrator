
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .compiler import ActionCompiler
from .retrieval import assemble_query_pool, pack_workspace, workspace_to_tokens
from .schemas import ActionIntent, SubgoalDecision, TargetSelection
from .utils import bbox_center, cosine, l2


ACTION_VOCAB = ["tap", "scroll", "drag", "type", "wait", "back", "dismiss", "explore", "key_combo", "type_text"]


# ── embedding-grounded task understanding ────────────────────────────
#
# No regex. No verb maps. No hardcoded splits.
#
# The task text and every visible node's text both live in the same
# embedding space (TextEncoder). Task understanding reduces to:
#
#   1. Embed the full task string.
#   2. For each visible node, compute similarity between the node's
#      text embedding and the task embedding.
#   3. The best-matching node is the target.
#   4. The node's role (from the belief graph) determines the verb.
#   5. If the node is a text field and the task contains words NOT
#      matching any visible element, those unmatched words are likely
#      the text to type.
#
# Multi-step tracking: once a node has been successfully interacted
# with (verifier says success), suppress it in future scoring. The
# next-best node becomes the target. This naturally sequences
# multi-step tasks without any instruction parsing.


class TaskWorker:
    def __init__(self, compiler: ActionCompiler) -> None:
        self.compiler = compiler
        self._interacted_node_ids: set[int] = set()
        self._typed_texts: set[str] = set()

    def maybe_refresh_task_embedding(self, state: "RuntimeState") -> None:
        txt = state.task_state.active_goal or ""
        if txt != state._last_embedded_task_text:
            state.task_embedding = state.models.encode_text([txt])[0].cpu().tolist() if txt else [0.0] * 128
            state._last_embedded_task_text = txt
            # Reset interaction history when task changes
            self._interacted_node_ids.clear()
            self._typed_texts.clear()

    # ── track what has been done ─────────────────────────────────────

    def _mark_interaction(self, state: "RuntimeState") -> None:
        """Track successfully interacted nodes so we don't repeat them."""
        verdicts = state.event_journal.tail(30, {"verifier_judgment"})
        if not verdicts:
            return
        v = verdicts[-1].payload.get("verdict")
        if v is None:
            return
        label = v.label if hasattr(v, "label") else v.get("label", "")
        if label not in {"success", "partial"}:
            return

        actions = state.event_journal.tail(30, {"action_issued"})
        if not actions:
            return
        last = actions[-1].payload
        for nid in last.get("target_node_ids", []):
            self._interacted_node_ids.add(nid)
        # Track typed text
        chunk_kind = last.get("chunk_kind", "")
        if chunk_kind == "type_text":
            cmd = last.get("cmd", {})
            text = cmd.get("text", "")
            if text:
                self._typed_texts.add(text.lower())

    # Also detect completion from chunk clearing (for no-expectation actions)
    def _mark_chunk_completion(self, state: "RuntimeState") -> None:
        if state.active_action_chunk is None and state.pending_target is not None:
            target = state.pending_target
            for nid in getattr(target, "node_ids", []):
                self._interacted_node_ids.add(nid)

    # ── subgoal: just describe what we're doing ──────────────────────

    def infer_subgoal(self, state: "RuntimeState") -> SubgoalDecision:
        task = state.task_state.active_goal or ""
        if not task:
            return SubgoalDecision("no_task", 0.1)

        # If all high-scoring nodes have been interacted with, task may be done
        if len(self._interacted_node_ids) > 0:
            # Check if there are still unmatched task-relevant nodes
            unmatched = self._find_unmatched_task_tokens(state)
            if not unmatched:
                return SubgoalDecision("task_may_be_complete", 0.7)

        return SubgoalDecision("pursue_task", 0.8)

    def _find_unmatched_task_tokens(self, state: "RuntimeState") -> list[str]:
        """
        Find task tokens not covered by any interacted node or typed text.
        Uses embedding similarity, not stop word lists.
        """
        task = (state.task_state.active_goal or "").lower()
        task_toks = [t for t in task.split() if len(t) > 2]
        if not task_toks:
            return []

        # Collect interacted node texts
        covered_texts: list[str] = []
        for nid in self._interacted_node_ids:
            if nid not in state.graph.nodes:
                continue
            node = state.graph.node(nid)
            t = (node.state.get("text") or "").lower()
            if t:
                covered_texts.append(t)
        for typed in self._typed_texts:
            covered_texts.append(typed)

        if not covered_texts:
            return task_toks

        # Embed task tokens and covered texts
        all_texts = task_toks + covered_texts
        all_embs = state.models.encode_text(all_texts)
        n_task = len(task_toks)
        task_embs = all_embs[:n_task]
        covered_embs = all_embs[n_task:]

        # Tokens with high similarity to covered texts are "done"
        sims = F.cosine_similarity(
            task_embs.unsqueeze(1),
            covered_embs.unsqueeze(0),
            dim=-1,
        )
        max_sims = sims.max(dim=1).values

        unmatched = []
        for i, tok in enumerate(task_toks):
            if float(max_sims[i].item()) < 0.5:
                unmatched.append(tok)
        return unmatched

    # ── target selection: embedding similarity, not keyword parsing ──

    def choose_target(self, state: "RuntimeState", workspace) -> Optional[TargetSelection]:
        if not workspace.nodes:
            return None

        task = state.task_state.active_goal or ""
        if not task:
            return None

        task_vec = state.task_embedding
        task_lower = task.lower()

        # ── build scoring tensors for NodeScorer ──
        # The NodeScorer computes f(z_obj_i, z_context, extra_features)
        # Context = task embedding fused with z_focus + z_world
        import torch

        node_vecs = []
        extra_vecs = []
        valid_nodes = []

        for wn in workspace.nodes:
            node = state.graph.node(wn.node_id)
            bb = wn.bbox or node.state.get("bbox")
            if bb is None:
                continue

            # z_obj (128-d canonical embedding — now fused with text/structure/role)
            z = node.z_obj or [0.0] * 128
            z = z[:128] + [0.0] * max(0, 128 - len(z))
            node_vecs.append(z)

            # Extra features: actionability, text overlap, interaction history, spatial
            role_probs = node.state.get("role_probs", {})
            text = (node.state.get("text") or "").lower()

            # Text overlap with task
            text_toks = set(t for t in text.replace("/", " ").replace("_", " ").split() if len(t) > 1)
            task_toks = set(t for t in task_lower.replace("/", " ").replace("_", " ").split() if len(t) > 1)
            overlap = len(text_toks & task_toks) / max(len(task_toks), 1) if text_toks and task_toks else 0.0

            cx, cy = bbox_center(bb)
            d = l2((cx, cy), (state.pointer.x_hat, state.pointer.y_hat))

            extra = [
                node.confidence,
                role_probs.get("ActionableDiscrete", 0.0),
                role_probs.get("TextEntry", 0.0),
                overlap,
                1.0 if wn.node_id in self._interacted_node_ids else 0.0,
                min(1.0, d / 1000.0),
                float(node.scale) / 5.0,
                1.0 if node.kind in {"container", "screen_region", "historic_anchor", "failure_pattern"} else 0.0,
            ]
            extra_vecs.append(extra)
            valid_nodes.append(wn.node_id)

        if not valid_nodes:
            return None

        # ── model-based scoring via NodeScorer ──
        node_t = torch.tensor(node_vecs, dtype=torch.float32)
        extra_t = torch.tensor(extra_vecs, dtype=torch.float32)

        # Context vector: task embedding (conditioned read from spec)
        ctx = task_vec[:128] if task_vec else [0.0] * 128
        ctx_t = torch.tensor(ctx, dtype=torch.float32)

        model_scores = state.models.score_nodes(node_t, ctx_t, extra_t)  # (N,)

        # ── combine model scores with embedding similarity ──
        scored = []
        for i, nid in enumerate(valid_nodes):
            node = state.graph.node(nid)
            ms = float(model_scores[i].item())

            # Embedding similarity (cosine of z_obj with task)
            emb_sim = cosine(node.z_obj[:128] if node.z_obj else [], task_vec[:128] if task_vec else []) if node.z_obj else 0.0

            # Combine: model score + embedding similarity + text overlap
            score = 0.3 * ms + 1.2 * max(0.0, emb_sim) + 0.8 * extra_vecs[i][3]

            # Suppress already-interacted
            if nid in self._interacted_node_ids:
                score *= 0.1

            # Suppress structural nodes
            if node.kind in {"container", "screen_region", "historic_anchor", "failure_pattern", "episode_summary"}:
                score *= 0.2

            scored.append((nid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score = scored[0]

        if best_score < 0.01:
            return None

        node = state.graph.node(best_id)
        bb = node.state.get("bbox")
        if bb is None:
            return None
        cx, cy = bbox_center(bb)
        return TargetSelection(
            node_ids=[best_id],
            target_distribution={"mean": (cx, cy), "bbox": bb, "radius": 8.0},
            confidence=min(1.0, max(0.1, best_score)),
        )

    # ── intent: derived from node role, not from parsing verbs ───────

    def choose_intent(self, state: "RuntimeState", target: Optional[TargetSelection], workspace) -> ActionIntent:
        if target is None:
            return ActionIntent(kind="wait", params={}, confidence=0.4)

        node = state.graph.node(target.node_ids[0])
        role_probs = node.state.get("role_probs", {})
        text = (node.state.get("text") or "").lower()

        # ── if this is a text field, figure out what to type ──
        if role_probs.get("TextEntry", 0.0) > 0.3 and state.body.supports_keyboard:
            type_text = self._infer_text_to_type(state, node)
            if type_text:
                return ActionIntent(kind="type_text", params={"text": type_text}, confidence=0.85)
            # Fall through to tap (to focus the field)

        # ── if the target node looks like a "back" affordance ──
        if text in {"back", "←", "<", "go back", "return", "previous"}:
            if state.body.supports_keyboard:
                return ActionIntent(kind="key_combo", params={"key": "ArrowLeft", "modifiers": ["Alt"]}, confidence=0.8)
            return ActionIntent(kind="back", params={}, confidence=0.8)

        # ── if scrollable container ──
        if role_probs.get("HierarchicalContainer", 0.0) > 0.5:
            return ActionIntent(kind="scroll", params={"direction": "down", "amount": 0.5}, confidence=0.7)

        # ── default: tap ──
        return ActionIntent(kind="tap", params={}, confidence=0.85)

    def _infer_text_to_type(self, state: "RuntimeState", target_node) -> str:
        """
        Figure out what to type by per-token embedding contrast.

        1. Embed each task token individually.
        2. Embed each visible element's text.
        3. For each task token, compute max similarity to any visible text.
        4. Tokens with LOW similarity to visible elements are "content"
           (things to type, not things to click).
        5. Find the best contiguous run of content tokens.
        """
        task = (state.task_state.active_goal or "").lower()
        if not task:
            return ""

        visible_texts: list[str] = []
        for nid in state.residency.resident_ids:
            n = state.graph.node(nid)
            t = (n.state.get("text") or "").strip()
            if t and len(t) > 1:
                visible_texts.append(t.lower())

        task_tokens = task.split()
        if not task_tokens or not visible_texts:
            return ""

        # Embed each token and each visible text
        all_texts = task_tokens + visible_texts
        all_embs = state.models.encode_text(all_texts)
        n_tok = len(task_tokens)
        tok_embs = all_embs[:n_tok]
        vis_embs = all_embs[n_tok:]

        # Per-token max similarity to visible elements
        sims = F.cosine_similarity(tok_embs.unsqueeze(1), vis_embs.unsqueeze(0), dim=-1)
        max_sims = sims.max(dim=1).values  # (n_tok,)

        # Mark each token as "content" (low vis similarity) or "reference" (high)
        threshold = 0.55
        is_content = [float(max_sims[i].item()) < threshold and len(task_tokens[i]) > 1
                      for i in range(n_tok)]

        # Find the longest contiguous run of content tokens
        best_span = ""
        best_len = 0
        i = 0
        while i < n_tok:
            if not is_content[i]:
                i += 1
                continue
            start = i
            while i < n_tok and is_content[i]:
                i += 1
            span = " ".join(task_tokens[start:i])
            if len(span) > best_len and span not in self._typed_texts:
                best_span = span
                best_len = len(span)

        return best_span.strip()

    # ── main tick ────────────────────────────────────────────────────

    def tick(self, state: "RuntimeState", scale_level: int = 4) -> None:
        self.maybe_refresh_task_embedding(state)
        self._mark_interaction(state)
        self._mark_chunk_completion(state)

        q = assemble_query_pool(state, state.complexes["task"], scale_level)
        ws = pack_workspace(state, q)

        state.current_subgoal = self.infer_subgoal(state)
        target = self.choose_target(state, ws)
        intent = self.choose_intent(state, target, ws)

        state.interaction_state.candidate_target_ids = [n.node_id for n in ws.nodes]
        if target is not None:
            state.pending_target = target
            state.interaction_state.target_distribution = target.target_distribution
        state.pending_intent = intent

        if state.active_action_chunk is None:
            if target is None:
                return
            chunk = self.compiler.compile(intent, target, state)
            state.active_action_chunk = chunk
            state.gesture_state.active_chunk_id = chunk.id
