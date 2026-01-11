from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from ..serialization import load_session_bundle

LABEL_TO_ID = {'success': 0, 'partial': 1, 'failure': 2, 'ambiguous': 3, 'delayed': 4}


def _latest_metric_before(events, idx: int) -> Optional[Any]:
    for j in range(idx, -1, -1):
        if events[j].type == 'metric':
            return events[j]
    return None


def _metric_to_vec(metric_event: Any) -> torch.Tensor:
    payload = metric_event.payload if metric_event is not None else {}
    vals = [
        float(payload.get('pointer_uncertainty', 0.0)),
        float(payload.get('branch_entropy', 0.0)),
        float(payload.get('fragile_action_phase', 0.0)),
        float(payload.get('pending_timeout_pressure', 0.0)),
        float(payload.get('failure_density', 0.0)),
        float(payload.get('ambiguity_score', 0.0)),
        float(payload.get('analogy_match_score', 0.0)),
        float(payload.get('num_events', 0.0) / 1000.0),
        float(payload.get('num_nodes', 0.0) / 1000.0),
    ]
    vals += [0.0] * (128 - len(vals))
    return torch.tensor(vals, dtype=torch.float32)


def _pad(vec: list[float] | None, d: int) -> list[float]:
    if vec is None:
        return [0.0] * d
    return vec[:d] + [0.0] * max(0, d - len(vec))


def _node_feature(node: Any, scale_level: int) -> tuple[torch.Tensor, torch.Tensor]:
    node_vec = torch.tensor(_pad(node.z_obj, 128), dtype=torch.float32)
    extra = torch.tensor([
        float(node.confidence),
        float(node.scale),
        float(scale_level),
        float(node.state.get('actionable_prob', 0.0)),
        float(node.state.get('selected', False)),
        float(len(node.state.get('text', '') or '')),
        0.0,
        0.0,
    ], dtype=torch.float32)
    return node_vec, extra


class VerifierDataset(Dataset):
    def __init__(self, examples: list[tuple[torch.Tensor, int]]) -> None:
        self.examples = examples

    @classmethod
    def from_directory(cls, root: str | Path) -> 'VerifierDataset':
        root = Path(root)
        examples = []
        for p in sorted(root.glob('**/bundle.pkl')):
            bundle = load_session_bundle(p)
            events = bundle['events']
            for i, ev in enumerate(events):
                if ev.type != 'verifier_judgment':
                    continue
                metric = _latest_metric_before(events, i)
                x = _metric_to_vec(metric)
                verdict = ev.payload['verdict']
                label = verdict.label if hasattr(verdict, 'label') else verdict['label']
                y = LABEL_TO_ID[label]
                examples.append((x, y))
        return cls(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


class TargetSelectionDataset(Dataset):
    def __init__(self, examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        self.examples = examples

    @classmethod
    def from_directory(cls, root: str | Path, negatives_per_positive: int = 1) -> 'TargetSelectionDataset':
        root = Path(root)
        examples = []
        rng = random.Random(7)
        for p in sorted(root.glob('**/bundle.pkl')):
            bundle = load_session_bundle(p)
            events = bundle['events']
            nodes = {n.id: n for n in bundle['nodes']}
            node_ids = list(nodes.keys())
            for i, ev in enumerate(events):
                if ev.type != 'action_issued' or ev.payload.get('phase') != 'approach':
                    continue
                targets = ev.payload.get('target_node_ids', [])
                if not targets:
                    continue
                pos_id = targets[0]
                if pos_id not in nodes:
                    continue
                metric = _latest_metric_before(events, i)
                metric_vec = _metric_to_vec(metric)
                pos_vec, pos_extra = _node_feature(nodes[pos_id], scale_level=nodes[pos_id].scale)
                negatives = [nid for nid in node_ids if nid != pos_id]
                rng.shuffle(negatives)
                for neg_id in negatives[:negatives_per_positive]:
                    neg_vec, neg_extra = _node_feature(nodes[neg_id], scale_level=nodes[neg_id].scale)
                    examples.append((metric_vec, pos_vec, pos_extra, neg_vec, neg_extra))
        return cls(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]
