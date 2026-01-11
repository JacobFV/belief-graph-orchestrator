
from __future__ import annotations

from typing import Any, Optional

from ..target import GUITarget
from ..schemas import FramePacket


class SimulatorPhone(GUITarget):
    """
    Backend for a mobile device simulator.

    TODO: implement using ``xcrun simctl`` for device control and
    ``xcrun simctl io screenshot`` for frame capture.
    """

    def __init__(
        self,
        key: str,
        width: int = 320,
        height: int = 640,
        device_udid: str | None = None,
    ) -> None:
        super().__init__(key, width, height)
        self.device_udid = device_udid

    def get_new_frame(self) -> Optional[FramePacket]:
        raise NotImplementedError("SimulatorPhone.get_new_frame is not yet implemented")

    def send_hid(self, vx: float, vy: float, contact: bool, button_mask: int = 1) -> None:
        raise NotImplementedError("SimulatorPhone.send_hid is not yet implemented")

    def get_hid_ack(self) -> Optional[dict]:
        raise NotImplementedError("SimulatorPhone.get_hid_ack is not yet implemented")

    def get_task_instruction(self) -> Optional[str]:
        raise NotImplementedError("SimulatorPhone.get_task_instruction is not yet implemented")
