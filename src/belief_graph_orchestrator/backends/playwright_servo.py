
from __future__ import annotations

import io
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch

from ..target import GUITarget
from ..schemas import FramePacket
from ..utils import clip

# JS injected into any page to add a servo-controlled cursor overlay.
# The cursor moves via velocity integration — no privileged mouse API.
# Click fires via elementFromPoint when contact transitions high→low.
_SERVO_CURSOR_JS = """
(() => {
  if (window.__servoCursorReady) return;
  window.__servoCursorReady = true;

  const GAIN = 20.0;
  let px = window.innerWidth / 2;
  let py = window.innerHeight / 2;
  let contact = false;
  let lastContact = false;

  // Create visible cursor element
  const cursor = document.createElement('div');
  cursor.id = '__servo_cursor';
  cursor.style.cssText = `
    position: fixed; z-index: 2147483647; pointer-events: none;
    width: 16px; height: 16px; transform: translate(-50%, -50%);
    border-radius: 50%; border: 3px solid #ff2222;
    background: rgba(255, 80, 80, 0.35);
    box-shadow: 0 0 6px rgba(255,0,0,0.5);
    left: ${px}px; top: ${py}px;
    transition: none;
  `;
  document.body.appendChild(cursor);

  // Crosshair arms
  const hLine = document.createElement('div');
  hLine.style.cssText = `
    position: fixed; z-index: 2147483647; pointer-events: none;
    width: 24px; height: 2px; background: #ff2222;
    transform: translate(-50%, -50%);
    left: ${px}px; top: ${py}px;
  `;
  document.body.appendChild(hLine);
  const vLine = document.createElement('div');
  vLine.style.cssText = `
    position: fixed; z-index: 2147483647; pointer-events: none;
    width: 2px; height: 24px; background: #ff2222;
    transform: translate(-50%, -50%);
    left: ${px}px; top: ${py}px;
  `;
  document.body.appendChild(vLine);

  function updateCursorDOM() {
    cursor.style.left = px + 'px';
    cursor.style.top  = py + 'px';
    hLine.style.left  = px + 'px';
    hLine.style.top   = py + 'px';
    vLine.style.left  = px + 'px';
    vLine.style.top   = py + 'px';
    if (contact) {
      cursor.style.background = 'rgba(255, 80, 80, 0.7)';
      cursor.style.borderColor = '#ff0000';
    } else {
      cursor.style.background = 'rgba(255, 80, 80, 0.35)';
      cursor.style.borderColor = '#ff2222';
    }
  }

  window.__servoCursor = {
    sendHID: function(vx, vy, c, buttonMask) {
      px = Math.max(0, Math.min(window.innerWidth - 1, px + GAIN * vx));
      py = Math.max(0, Math.min(window.innerHeight - 1, py + GAIN * vy));
      lastContact = contact;
      contact = c;
      updateCursorDOM();
      // Click on release (contact high→low)
      if (lastContact && !contact) {
        // Hide cursor briefly so elementFromPoint doesn't hit it
        cursor.style.display = 'none';
        hLine.style.display = 'none';
        vLine.style.display = 'none';
        const el = document.elementFromPoint(px, py);
        cursor.style.display = '';
        hLine.style.display = '';
        vLine.style.display = '';
        if (el) {
          el.focus();
          el.click();
        }
      }
    },
    getPos: function() { return [px, py]; },
  };
})();
"""


class PlaywrightServoTarget(GUITarget):
    """
    Playwright backend with servo-controlled cursor — no privileged positioning.

    The cursor is a visible overlay injected into the page.  It moves via
    velocity commands (send_hid) and clicks via DOM elementFromPoint on
    contact release.  Element detection comes purely from vision — no DOM
    queries for bounding boxes.

    Capability flags:
      has_direct_cursor    = False  (must detect cursor from pixels)
      supports_absolute_move = False  (must servo to targets)
      supports_keyboard    = True   (keyboard input is direct)
    """

    @property
    def supports_keyboard(self) -> bool:
        return True

    def __init__(
        self,
        key: str,
        width: int = 1280,
        height: int = 800,
        *,
        headless: bool = True,
        url: str = "https://example.com",
        task: str = "",
    ) -> None:
        super().__init__(key, width, height)
        self._headless = headless
        self._url = url
        self._task_queue: list[str] = [task] if task else []
        self._last_ack: Optional[dict] = None
        self._contact = False
        self._last_contact = False
        self._pw_context_manager: Any = None
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._started = False
        self._frame_counter = 0

    # ── lifecycle ────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        if self._started:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "PlaywrightServoTarget requires 'playwright'. "
                "Install with: pip install playwright && python -m playwright install chromium"
            ) from exc

        self._pw_context_manager = sync_playwright()
        self._playwright = self._pw_context_manager.start()
        self._browser = self._playwright.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page(viewport={"width": self.width, "height": self.height})
        self._page.goto(self._url, wait_until="domcontentloaded", timeout=15000)
        self._page.wait_for_timeout(500)
        self._page.evaluate(_SERVO_CURSOR_JS)
        self._started = True

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._pw_context_manager.__exit__(None, None, None)
            self._playwright = None
        self._started = False

    def __del__(self) -> None:
        self.close()

    # ── GUITarget interface ──────────────────────────────────────────

    def get_new_frame(self) -> Optional[FramePacket]:
        self._ensure_started()
        assert self._page is not None
        self._frame_counter += 1

        screenshot_bytes: bytes = self._page.screenshot()
        tensor = self._png_bytes_to_tensor(screenshot_bytes)

        # Minimal metadata — NO element bounding boxes.
        # Element detection must come from vision (the perception worker's
        # fallback path: grid proposals + pixel analysis).
        title = self._page.title() or ""
        metadata: dict[str, Any] = {
            "screen_id": title[:40] or self._url[:40],
            "frame_idx": self._frame_counter,
            "page_title": title,
            "page_url": self._page.url,
            # pointer_hint: the servo JS tracks this — it's fair game since
            # we injected the cursor ourselves (analogous to knowing what
            # velocity commands we sent).  The brain's Kalman filter can
            # still use visual detection as the primary signal.
            "pointer_hint": tuple(self._page.evaluate("window.__servoCursor.getPos()")),
        }
        return FramePacket(image=tensor, t_capture_ns=time.time_ns(), metadata=metadata)

    def send_hid(self, vx: float, vy: float, contact: bool, button_mask: int = 1) -> None:
        self._ensure_started()
        assert self._page is not None
        contact_js = "true" if contact else "false"
        self._page.evaluate(
            f"window.__servoCursor.sendHID({vx}, {vy}, {contact_js}, {button_mask})"
        )
        self._last_ack = {
            "vx": vx, "vy": vy, "contact": contact,
            "button_mask": button_mask, "t_ns": time.time_ns(),
        }

    def send_key(self, key: str, modifiers: list[str] | None = None) -> None:
        self._ensure_started()
        assert self._page is not None
        combo = "+".join((modifiers or []) + [key])
        self._page.keyboard.press(combo)
        self._last_ack = {"type": "key", "key": key, "modifiers": modifiers or [], "t_ns": time.time_ns()}

    def send_text(self, text: str) -> None:
        self._ensure_started()
        assert self._page is not None
        self._page.keyboard.type(text, delay=20)
        self._last_ack = {"type": "text", "text": text, "t_ns": time.time_ns()}

    def get_hid_ack(self) -> Optional[dict]:
        ack = self._last_ack
        self._last_ack = None
        return ack

    def get_task_instruction(self) -> Optional[str]:
        if self._task_queue:
            return self._task_queue.pop(0)
        return None

    # ── helpers ──────────────────────────────────────────────────────

    def _png_bytes_to_tensor(self, data: bytes) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Requires Pillow: pip install Pillow") from exc
        import numpy as np
        img = Image.open(io.BytesIO(data)).convert("RGB").resize((self.width, self.height))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
