from __future__ import annotations

from typing import Any, Optional

from ..target import GUITarget
from ..runtime import Brain
from ..schemas import FramePacket

class ReplayTarget(GUITarget):
    def __init__(self, key: str, bundle: dict[str, Any]) -> None:
        super().__init__(key)
        self.events = list(bundle.get('events', []))
        self._cursor = 0
        self._prepared = False
        self._pending_task: Optional[str] = None
        self._pending_ack: Optional[dict[str, Any]] = None
        self._pending_frame: Optional[FramePacket] = None
        self._sent_cmds: list[dict[str, Any]] = []

    def _prepare(self) -> None:
        if self._prepared:
            return
        self._pending_task = None
        self._pending_ack = None
        self._pending_frame = None
        while self._cursor < len(self.events):
            ev = self.events[self._cursor]
            self._cursor += 1
            if ev.type == 'task_instruction' and self._pending_task is None:
                self._pending_task = ev.payload.get('text')
            elif ev.type == 'hid_ack' and self._pending_ack is None:
                self._pending_ack = ev.payload.get('ack')
            elif ev.type == 'frame':
                fp = ev.payload.get('frame_packet')
                if fp is None:
                    fp = FramePacket(image=ev.payload['image'], t_capture_ns=ev.t_capture_ns, metadata=ev.payload.get('metadata', {}))
                self._pending_frame = fp
                break
        self._prepared = True

    def get_task_instruction(self) -> Optional[str]:
        self._prepare()
        out = self._pending_task
        self._pending_task = None
        return out

    def get_hid_ack(self) -> Optional[dict]:
        self._prepare()
        out = self._pending_ack
        self._pending_ack = None
        return out

    def get_new_frame(self) -> Optional[FramePacket]:
        self._prepare()
        out = self._pending_frame
        self._pending_frame = None
        self._prepared = False
        return out

    def send_hid(self, vx: float, vy: float, contact: bool, button_mask: int = 1) -> None:
        self._sent_cmds.append({'vx': vx, 'vy': vy, 'contact': contact, 'button_mask': button_mask})


def replay_bundle(bundle: dict[str, Any], steps: int = 120) -> dict[str, Any]:
    phone = ReplayTarget('replay-key', bundle)
    brain = Brain('replay-key', target_instance=phone, use_metadata_hints=True)
    for _ in range(steps):
        brain.step()
    return brain.summary()
