from __future__ import annotations

import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return {"_kind": "tensor", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if is_dataclass(obj):
        return _sanitize(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def bundle_from_runtime_state(state: Any) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "episode_id": state.episode_id,
        "summary": {
            "task": state.task_state.active_goal,
            "screen": state.latest_screen_id,
            "num_events": len(state.event_journal.events),
            "num_nodes": len(state.graph.nodes),
            "num_edges": len(state.graph.edges),
            "num_branches": len(state.branches),
        },
        "events": list(state.event_journal.events),
        "nodes": list(state.graph.nodes.values()),
        "edges": list(state.graph.edges),
        "expectations": dict(state.expectations),
        "branches": dict(state.branches),
    }


def save_session_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'bundle.pkl', 'wb') as f:
        pickle.dump(bundle, f)
    with open(path / 'summary.json', 'w') as f:
        json.dump(_sanitize(bundle['summary']), f, indent=2)
    with open(path / 'events.jsonl', 'w') as f:
        for ev in bundle['events']:
            f.write(json.dumps(_sanitize(ev)) + '\n')


def load_session_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    bundle_path = path / 'bundle.pkl' if path.is_dir() else path
    with open(bundle_path, 'rb') as f:
        return pickle.load(f)
