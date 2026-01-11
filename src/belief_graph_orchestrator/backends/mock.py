
from __future__ import annotations

import math
import time
from typing import Any, Optional

import torch

from ..target import GUITarget, UIElement
from ..schemas import FramePacket
from ..utils import clip


class MockPhone(GUITarget):
    """
    Very small deterministic mock surface so the brain can run end-to-end.
    """
    def __init__(self, key: str, width: int = 320, height: int = 640) -> None:
        super().__init__(key, width, height)
        self.pointer_x = width * 0.5
        self.pointer_y = height * 0.5
        self.contact = False
        self._last_contact = False
        self._last_ack: Optional[dict] = None
        self._task_queue: list[str] = ["open settings and toggle dark mode"]
        self.current_screen = "home"
        self.dark_mode = False
        self.compose_buffer = ""
        self.scroll_offset = 0.0
        self._frame_counter = 0

    def get_task_instruction(self) -> Optional[str]:
        if self._task_queue:
            return self._task_queue.pop(0)
        return None

    def get_hid_ack(self) -> Optional[dict]:
        ack = self._last_ack
        self._last_ack = None
        return ack

    def send_hid(self, vx: float, vy: float, contact: bool, button_mask: int = 1) -> None:
        gain = 20.0
        self.pointer_x = clip(self.pointer_x + gain * vx, 0, self.width - 1)
        self.pointer_y = clip(self.pointer_y + gain * vy, 0, self.height - 1)

        self._last_contact = self.contact
        self.contact = contact

        if self._last_contact and not self.contact:
            self._handle_click()

        self._last_ack = {
            "vx": vx,
            "vy": vy,
            "contact": contact,
            "button_mask": button_mask,
            "t_ns": time.time_ns(),
        }

    def get_new_frame(self) -> Optional[FramePacket]:
        self._frame_counter += 1
        img = self._render_screen()
        metadata = {
            "screen_id": self.current_screen,
            "elements": [
                {
                    "id": e.id,
                    "text": e.text,
                    "bbox": e.bbox,
                    "role": e.role,
                    "selected": e.selected,
                    "enabled": e.enabled,
                    "action": e.action,
                }
                for e in self._elements_for_screen()
            ],
            "pointer_hint": (self.pointer_x, self.pointer_y),
            "dark_mode": self.dark_mode,
            "frame_idx": self._frame_counter,
        }
        return FramePacket(image=img, t_capture_ns=time.time_ns(), metadata=metadata)

    def _elements_for_screen(self) -> list[UIElement]:
        W, H = self.width, self.height
        if self.current_screen == "home":
            return [
                UIElement("title", "home", (20, 20, W - 20, 80), "label"),
                UIElement("settings", "open settings", (40, 120, W - 40, 180), "button",
                          {"type": "goto", "screen": "settings"}),
                UIElement("compose", "compose", (40, 220, W - 40, 280), "button",
                          {"type": "goto", "screen": "compose"}),
                UIElement("messages", "messages", (40, 320, W - 40, 380), "button",
                          {"type": "goto", "screen": "messages"}),
            ]
        if self.current_screen == "settings":
            return [
                UIElement("back", "back", (16, 16, 96, 64), "button", {"type": "goto", "screen": "home"}),
                UIElement("title", "settings", (110, 16, W - 16, 64), "label"),
                UIElement("toggle", "dark mode", (40, 140, W - 40, 220), "toggle",
                          {"type": "toggle_dark"}, selected=self.dark_mode),
            ]
        if self.current_screen == "compose":
            return [
                UIElement("back", "back", (16, 16, 96, 64), "button", {"type": "goto", "screen": "home"}),
                UIElement("title", "compose", (110, 16, W - 16, 64), "label"),
                UIElement("textfield", "message", (30, 120, W - 30, 220), "text_field",
                          {"type": "focus"}),
                UIElement("send", "send", (W - 120, H - 100, W - 20, H - 40), "button",
                          {"type": "goto", "screen": "home"}),
            ]
        if self.current_screen == "messages":
            return [
                UIElement("back", "back", (16, 16, 96, 64), "button", {"type": "goto", "screen": "home"}),
                UIElement("title", "messages", (110, 16, W - 16, 64), "label"),
                UIElement("msg1", "hello", (20, 120, W - 20, 180), "list_item"),
                UIElement("msg2", "world", (20, 200, W - 20, 260), "list_item"),
            ]
        return []

    def _handle_click(self) -> None:
        for e in self._elements_for_screen():
            if self._inside(self.pointer_x, self.pointer_y, e.bbox) and e.enabled:
                act = e.action
                t = act.get("type")
                if t == "goto":
                    self.current_screen = act["screen"]
                    return
                if t == "toggle_dark":
                    self.dark_mode = not self.dark_mode
                    return
                if t == "focus":
                    return

    @staticmethod
    def _inside(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def _render_screen(self) -> torch.Tensor:
        bg = 0.08 if self.dark_mode else 0.92
        img = torch.full((3, self.height, self.width), bg, dtype=torch.float32)
        for e in self._elements_for_screen():
            x1, y1, x2, y2 = map(int, e.bbox)
            if e.role == "label":
                color = torch.tensor([0.2, 0.2, 0.2]) if not self.dark_mode else torch.tensor([0.85, 0.85, 0.85])
            elif e.role == "toggle":
                color = torch.tensor([0.2, 0.8, 0.2]) if e.selected else torch.tensor([0.8, 0.2, 0.2])
            elif e.role == "text_field":
                color = torch.tensor([0.7, 0.7, 1.0])
            else:
                color = torch.tensor([0.3, 0.55, 0.95])
            img[:, y1:y2, x1:x2] = color.view(3, 1, 1)

        # pointer as white dot with black halo
        px, py = int(self.pointer_x), int(self.pointer_y)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                xx, yy = px + dx, py + dy
                if 0 <= xx < self.width and 0 <= yy < self.height:
                    d = math.sqrt(dx * dx + dy * dy)
                    if d <= 4.0:
                        img[:, yy, xx] = torch.tensor([1.0, 1.0, 1.0]) if d <= 2.0 else torch.tensor([0.0, 0.0, 0.0])
        return img
