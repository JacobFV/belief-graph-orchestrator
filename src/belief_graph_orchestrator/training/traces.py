from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..serialization import load_session_bundle

@dataclass
class TraceIndexEntry:
    path: Path
    session_id: str
    num_events: int
    num_nodes: int

class TraceIndex:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def iter_entries(self) -> Iterator[TraceIndexEntry]:
        for p in sorted(self.root.glob('**/bundle.pkl')):
            bundle = load_session_bundle(p)
            yield TraceIndexEntry(
                path=p.parent,
                session_id=bundle.get('session_id', p.parent.name),
                num_events=len(bundle.get('events', [])),
                num_nodes=len(bundle.get('nodes', [])),
            )
