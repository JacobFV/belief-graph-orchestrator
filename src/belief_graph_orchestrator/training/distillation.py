"""
Pseudo-AX structure distillation: collect privileged teacher data from
backends that expose DOM/accessibility structure, then use it to train
a vision-only student model.

The spec says:
  "use privileged structure at training time to produce an unprivileged
   inference model at runtime."

This module provides:
  - TeacherCollector: extracts structured element data from privileged
    backends (PlaywrightDesktop with DOM access) and emits teacher_structure
    events into the event journal.
  - collect_distillation_pairs: runs a brain session with privileged hints
    and collects (screenshot, structure) pairs for training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime import Brain
from ..schemas import FramePacket


class TeacherCollector:
    """
    Collects teacher_structure events by extracting DOM elements from a
    privileged backend during training runs.

    Usage:
        collector = TeacherCollector(brain)
        # During each step:
        collector.maybe_collect(brain)
    """

    def __init__(self, interval: int = 5) -> None:
        self._interval = interval
        self._step = 0

    def maybe_collect(self, brain: Brain) -> None:
        """If the latest frame has element metadata, emit a teacher_structure event."""
        self._step += 1
        if self._step % self._interval != 0:
            return

        frame_events = brain.journal.tail(20, {"frame"})
        if not frame_events:
            return
        fp = frame_events[-1].payload.get("frame_packet")
        if fp is None:
            return
        elements = fp.metadata.get("elements")
        if not elements:
            return

        # Emit teacher_structure event with privileged element annotations
        ev = brain.journal.make_event(
            "teacher_structure",
            brain.state.session_id,
            brain.state.episode_id,
            {
                "elements": elements,
                "screen_id": fp.metadata.get("screen_id"),
                "page_title": fp.metadata.get("page_title"),
                "page_url": fp.metadata.get("page_url"),
                "frame_idx": fp.metadata.get("frame_idx"),
                # Role labels derived from privileged metadata
                "role_labels": [
                    {
                        "id": el.get("id"),
                        "bbox": el.get("bbox"),
                        "role": el.get("role"),
                        "text": el.get("text"),
                        "enabled": el.get("enabled"),
                        "selected": el.get("selected"),
                    }
                    for el in elements
                ],
            },
            parent_ids=[frame_events[-1].id],
        )
        brain.journal.append(ev)


def collect_distillation_pairs(
    out_dir: str | Path,
    num_sessions: int = 4,
    steps_per_session: int = 60,
    backend: str = "mock-phone",
) -> None:
    """
    Run brain sessions with privileged metadata hints, collecting
    teacher_structure events alongside normal perception events.

    The resulting bundles contain paired (frame, teacher_structure)
    events that can be used to train a vision-only student model.
    """
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_sessions):
        if backend == "mock-desktop":
            from ..backends.mock_desktop import MockDesktop
            target = MockDesktop(f"teacher-{i}")
            brain = Brain(target_key=f"teacher-{i}", target_instance=target, use_metadata_hints=True)
        else:
            from ..backends.mock import MockPhone
            brain = Brain(target_key=f"teacher-{i}", target_cls=MockPhone, use_metadata_hints=True)

        collector = TeacherCollector(interval=3)

        for _ in range(steps_per_session):
            brain.step()
            collector.maybe_collect(brain)

        brain.save_bundle(str(out_dir / f"distill_{i:03d}"))

        # Also save a summary of teacher events
        teacher_events = [
            e for e in brain.journal.events if e.type == "teacher_structure"
        ]
        summary = {
            "session_id": brain.state.session_id,
            "num_teacher_events": len(teacher_events),
            "num_total_events": len(brain.journal.events),
            "num_nodes": len(brain.graph.nodes),
        }
        (out_dir / f"distill_{i:03d}_summary.json").write_text(json.dumps(summary, indent=2))
