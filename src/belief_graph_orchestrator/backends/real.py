
from __future__ import annotations

from typing import Any, Optional

from ..target import GUITarget
from ..schemas import FramePacket


class RealPhone(GUITarget):
    """
    Stub backend for a physical mobile device connected over USB/network.

    TODO: implement with your chosen transport (e.g. usbmuxd, WebDriverAgent,
    or a custom HID relay).
    """

    def __init__(self, key: str, width: int = 320, height: int = 640) -> None:
        super().__init__(key, width, height)

    def get_new_frame(self) -> Optional[FramePacket]:
        raise NotImplementedError("RealPhone.get_new_frame is not yet implemented")

    def send_hid(self, vx: float, vy: float, contact: bool, button_mask: int = 1) -> None:
        raise NotImplementedError("RealPhone.send_hid is not yet implemented")

    def get_hid_ack(self) -> Optional[dict]:
        raise NotImplementedError("RealPhone.get_hid_ack is not yet implemented")

    def get_task_instruction(self) -> Optional[str]:
        raise NotImplementedError("RealPhone.get_task_instruction is not yet implemented")
