
from __future__ import annotations

import math
import time
from typing import Any, Optional

import torch

from ..target import DesktopTarget, UIElement
from ..schemas import FramePacket
from ..utils import clip


class MockDesktop(DesktopTarget):
    """
    Deterministic mock desktop surface for testing.

    Provides a multi-screen desktop-like UI with buttons, text fields,
    checkboxes, and menus — exercising click, keyboard, and direct cursor
    paths so the Brain can run end-to-end without Playwright.
    """

    def __init__(self, key: str, width: int = 1280, height: int = 800) -> None:
        super().__init__(key, width, height)
        self.cursor_x: float = width / 2.0
        self.cursor_y: float = height / 2.0
        self._last_ack: Optional[dict] = None
        self._task_queue: list[str] = ["click the submit button and type hello"]
        self.current_screen = "main"
        self.text_buffer = ""
        self.checkbox_checked = False
        self.sidebar_open = True
        self._frame_counter = 0

    # ── capabilities (inherited True from DesktopTarget) ─────────────

    # ── observation ──────────────────────────────────────────────────

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
            "pointer_hint": (self.cursor_x, self.cursor_y),
            "frame_idx": self._frame_counter,
        }
        return FramePacket(image=img, t_capture_ns=time.time_ns(), metadata=metadata)

    def get_cursor_position(self) -> Optional[tuple[float, float]]:
        return (self.cursor_x, self.cursor_y)

    # ── motor commands ───────────────────────────────────────────────

    def move_cursor_to(self, x: float, y: float) -> None:
        self.cursor_x = clip(x, 0, self.width - 1)
        self.cursor_y = clip(y, 0, self.height - 1)
        self._last_ack = {"type": "move", "x": self.cursor_x, "y": self.cursor_y, "t_ns": time.time_ns()}

    def click(self, x: float, y: float, button: str = "left") -> None:
        self.cursor_x = clip(x, 0, self.width - 1)
        self.cursor_y = clip(y, 0, self.height - 1)
        self._handle_click()
        self._last_ack = {"type": "click", "x": self.cursor_x, "y": self.cursor_y, "button": button, "t_ns": time.time_ns()}

    def send_key(self, key: str, modifiers: list[str] | None = None) -> None:
        mods = modifiers or []
        if key == "Escape":
            if self.current_screen != "main":
                self.current_screen = "main"
        elif key == "ArrowLeft" and "Alt" in mods:
            if self.current_screen != "main":
                self.current_screen = "main"
        elif key == "Tab":
            pass  # focus cycling — no-op in mock
        self._last_ack = {"type": "key", "key": key, "modifiers": mods, "t_ns": time.time_ns()}

    def send_text(self, text: str) -> None:
        self.text_buffer += text
        self._last_ack = {"type": "text", "text": text, "t_ns": time.time_ns()}

    def get_hid_ack(self) -> Optional[dict]:
        ack = self._last_ack
        self._last_ack = None
        return ack

    def get_task_instruction(self) -> Optional[str]:
        if self._task_queue:
            return self._task_queue.pop(0)
        return None

    # ── screen definitions ───────────────────────────────────────────

    def _elements_for_screen(self) -> list[UIElement]:
        W, H = self.width, self.height
        if self.current_screen == "main":
            elems = [
                UIElement("menubar", "File  Edit  View  Help", (0, 0, W, 30), "menubar"),
                UIElement("title", "Desktop App", (W // 2 - 80, 40, W // 2 + 80, 70), "label"),
            ]
            if self.sidebar_open:
                elems += [
                    UIElement("sidebar_bg", "sidebar", (0, 30, 220, H), "sidebar"),
                    UIElement("nav_home", "Home", (10, 80, 210, 120), "button",
                              {"type": "goto", "screen": "main"}),
                    UIElement("nav_form", "Form", (10, 130, 210, 170), "button",
                              {"type": "goto", "screen": "form"}),
                    UIElement("nav_table", "Data Table", (10, 180, 210, 220), "button",
                              {"type": "goto", "screen": "table"}),
                ]
            content_left = 230 if self.sidebar_open else 10
            elems += [
                UIElement("welcome", "Welcome to the desktop app.", (content_left, 100, W - 20, 140), "label"),
                UIElement("submit", "Submit", (content_left, 200, content_left + 160, 250), "button",
                          {"type": "goto", "screen": "form"}),
                UIElement("toggle_sidebar", "Toggle Sidebar", (content_left, 280, content_left + 180, 320), "button",
                          {"type": "toggle_sidebar"}),
            ]
            return elems

        if self.current_screen == "form":
            return [
                UIElement("menubar", "File  Edit  View  Help", (0, 0, W, 30), "menubar"),
                UIElement("back", "Back", (10, 40, 80, 70), "button", {"type": "goto", "screen": "main"}),
                UIElement("title", "Form", (W // 2 - 40, 40, W // 2 + 40, 70), "label"),
                UIElement("name_label", "Name:", (40, 120, 140, 150), "label"),
                UIElement("name_field", self.text_buffer or "Enter name...", (150, 110, W - 40, 155), "text_field",
                          {"type": "focus"}),
                UIElement("checkbox", "Accept terms", (40, 200, 260, 240), "toggle",
                          {"type": "toggle_check"}, selected=self.checkbox_checked),
                UIElement("submit_btn", "Submit", (40, 300, 200, 350), "button",
                          {"type": "submit"}, enabled=self.checkbox_checked),
                UIElement("cancel_btn", "Cancel", (220, 300, 380, 350), "button",
                          {"type": "goto", "screen": "main"}),
            ]

        if self.current_screen == "table":
            rows = [
                UIElement("header", "ID  |  Name  |  Status", (40, 100, W - 40, 135), "label"),
            ]
            for i in range(6):
                y0 = 145 + i * 40
                rows.append(
                    UIElement(f"row_{i}", f"{i+1}  |  Item {i+1}  |  Active", (40, y0, W - 40, y0 + 35), "list_item")
                )
            return [
                UIElement("menubar", "File  Edit  View  Help", (0, 0, W, 30), "menubar"),
                UIElement("back", "Back", (10, 40, 80, 70), "button", {"type": "goto", "screen": "main"}),
                UIElement("title", "Data Table", (W // 2 - 60, 40, W // 2 + 60, 70), "label"),
            ] + rows

        return []

    def _handle_click(self) -> None:
        for e in self._elements_for_screen():
            if self._inside(self.cursor_x, self.cursor_y, e.bbox) and e.enabled:
                act = e.action
                t = act.get("type")
                if t == "goto":
                    self.current_screen = act["screen"]
                    return
                if t == "toggle_sidebar":
                    self.sidebar_open = not self.sidebar_open
                    return
                if t == "toggle_check":
                    self.checkbox_checked = not self.checkbox_checked
                    return
                if t == "submit":
                    self.current_screen = "main"
                    self.text_buffer = ""
                    self.checkbox_checked = False
                    return
                if t == "focus":
                    return

    @staticmethod
    def _inside(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    # ── rendering ────────────────────────────────────────────────────

    def _render_screen(self) -> torch.Tensor:
        img = torch.full((3, self.height, self.width), 0.94, dtype=torch.float32)

        COLOR_MAP = {
            "label": torch.tensor([0.15, 0.15, 0.15]),
            "button": torch.tensor([0.24, 0.47, 0.85]),
            "toggle": None,  # handled below
            "text_field": torch.tensor([1.0, 1.0, 1.0]),
            "list_item": torch.tensor([0.90, 0.92, 0.96]),
            "menubar": torch.tensor([0.22, 0.22, 0.24]),
            "sidebar": torch.tensor([0.88, 0.88, 0.90]),
        }

        for e in self._elements_for_screen():
            x1, y1, x2, y2 = map(int, e.bbox)
            x1, x2 = max(0, x1), min(self.width, x2)
            y1, y2 = max(0, y1), min(self.height, y2)
            if e.role == "toggle":
                color = torch.tensor([0.2, 0.7, 0.3]) if e.selected else torch.tensor([0.7, 0.2, 0.2])
            else:
                color = COLOR_MAP.get(e.role, torch.tensor([0.6, 0.6, 0.6]))
            img[:, y1:y2, x1:x2] = color.view(3, 1, 1)

        # draw cursor as crosshair
        px, py = int(self.cursor_x), int(self.cursor_y)
        for d in range(-6, 7):
            xx = px + d
            if 0 <= xx < self.width and 0 <= py < self.height:
                img[:, py, xx] = torch.tensor([1.0, 0.0, 0.0])
            yy = py + d
            if 0 <= px < self.width and 0 <= yy < self.height:
                img[:, yy, px] = torch.tensor([1.0, 0.0, 0.0])

        return img
