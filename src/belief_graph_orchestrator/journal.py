from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from .schemas import Event
from .utils import now_ns

class EventJournal:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._next_id = 1

    def next_id(self) -> int:
        eid = self._next_id
        self._next_id += 1
        return eid

    def append(self, event: Event) -> Event:
        self.events.append(event)
        self._next_id = max(self._next_id, event.id + 1)
        return event

    def make_event(self, event_type: str, session_id: str, episode_id: str, payload: dict, parent_ids: Optional[list[int]] = None, t_capture_ns: Optional[int] = None, uncertainty: Optional[dict[str, float]] = None) -> Event:
        return Event(
            id=self.next_id(),
            type=event_type,  # type: ignore[arg-type]
            t_capture_ns=t_capture_ns if t_capture_ns is not None else now_ns(),
            t_arrival_ns=now_ns(),
            session_id=session_id,
            episode_id=episode_id,
            parent_ids=parent_ids or [],
            payload=payload,
            uncertainty=uncertainty or {},
        )

    def get(self, event_id: int) -> Event:
        return self.events[event_id - 1]

    def tail(self, n: int, types: Optional[set[str]] = None) -> list[Event]:
        xs = self.events[-n:]
        return xs if types is None else [e for e in xs if e.type in types]

    def range(self, t0_ns: int, t1_ns: int, types: Optional[set[str]] = None) -> list[Event]:
        xs = [e for e in self.events if t0_ns <= e.t_capture_ns <= t1_ns]
        return xs if types is None else [e for e in xs if e.type in types]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.type] = counts.get(e.type, 0) + 1
        return counts

    def save(self, path: str | Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.events, f)

    @classmethod
    def load(cls, path: str | Path) -> 'EventJournal':
        with open(path, 'rb') as f:
            events = pickle.load(f)
        j = cls()
        for e in events:
            j.append(e)
        return j
