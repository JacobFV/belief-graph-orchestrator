
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

_MOCK_IOS_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    width: {WIDTH}px; height: {HEIGHT}px; overflow: hidden;
    font-family: -apple-system, 'Helvetica Neue', sans-serif;
  }
  #screen {
    width: 100%; height: 100%; position: relative;
  }
  .el {
    position: absolute;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 500; color: #fff;
    border-radius: 10px;
  }
  .el.label  { color: #222; background: none; font-weight: 700; font-size: 18px; }
  .el.button { background: #4d8df3; }
  .el.toggle { border-radius: 14px; }
  .el.toggle.on  { background: #34c759; }
  .el.toggle.off { background: #cc3333; }
  .el.text_field { background: #b3b3ff; color: #333; }
  .el.list_item  { background: #4d8df3; }
  body.dark .el.label { color: #ddd; }
  body.dark { background: #141414; }
  body:not(.dark) { background: #ebebeb; }
  #pointer {
    position: absolute; width: 10px; height: 10px;
    border-radius: 50%; background: #fff; border: 2px solid #000;
    z-index: 9999; pointer-events: none;
    transform: translate(-50%, -50%);
  }
</style>
</head>
<body>
<div id="screen"></div>
<div id="pointer"></div>
<script>
const W = {WIDTH}, H = {HEIGHT};
const GAIN = 20.0;

let currentScreen = 'home';
let darkMode = false;
let pointerX = W / 2;
let pointerY = H / 2;
let contact = false;
let lastContact = false;
let frameCounter = 0;

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

function inside(x, y, b) { return b[0] <= x && x <= b[2] && b[1] <= y && y <= b[3]; }

function getElements() {
  if (currentScreen === 'home') return [
    {id:'title',    text:'home',          bbox:[20,20,W-20,80],    role:'label',      action:{},                              selected:false, enabled:true},
    {id:'settings', text:'open settings', bbox:[40,120,W-40,180],  role:'button',     action:{type:'goto', screen:'settings'}, selected:false, enabled:true},
    {id:'compose',  text:'compose',       bbox:[40,220,W-40,280],  role:'button',     action:{type:'goto', screen:'compose'},  selected:false, enabled:true},
    {id:'messages', text:'messages',      bbox:[40,320,W-40,380],  role:'button',     action:{type:'goto', screen:'messages'}, selected:false, enabled:true},
  ];
  if (currentScreen === 'settings') return [
    {id:'back',   text:'back',      bbox:[16,16,96,64],    role:'button', action:{type:'goto', screen:'home'}, selected:false,  enabled:true},
    {id:'title',  text:'settings',  bbox:[110,16,W-16,64], role:'label',  action:{},                           selected:false,  enabled:true},
    {id:'toggle', text:'dark mode', bbox:[40,140,W-40,220],role:'toggle', action:{type:'toggle_dark'},          selected:darkMode, enabled:true},
  ];
  if (currentScreen === 'compose') return [
    {id:'back',      text:'back',    bbox:[16,16,96,64],          role:'button',     action:{type:'goto', screen:'home'}, selected:false, enabled:true},
    {id:'title',     text:'compose', bbox:[110,16,W-16,64],       role:'label',      action:{},                           selected:false, enabled:true},
    {id:'textfield', text:'message', bbox:[30,120,W-30,220],      role:'text_field', action:{type:'focus'},                selected:false, enabled:true},
    {id:'send',      text:'send',    bbox:[W-120,H-100,W-20,H-40],role:'button',     action:{type:'goto', screen:'home'}, selected:false, enabled:true},
  ];
  if (currentScreen === 'messages') return [
    {id:'back',  text:'back',     bbox:[16,16,96,64],    role:'button',    action:{type:'goto', screen:'home'}, selected:false, enabled:true},
    {id:'title', text:'messages', bbox:[110,16,W-16,64], role:'label',     action:{},                           selected:false, enabled:true},
    {id:'msg1',  text:'hello',    bbox:[20,120,W-20,180],role:'list_item', action:{},                           selected:false, enabled:true},
    {id:'msg2',  text:'world',    bbox:[20,200,W-20,260],role:'list_item', action:{},                           selected:false, enabled:true},
  ];
  return [];
}

function handleClick() {
  for (const e of getElements()) {
    if (inside(pointerX, pointerY, e.bbox) && e.enabled) {
      const t = e.action.type;
      if (t === 'goto')        { currentScreen = e.action.screen; renderScreen(); return; }
      if (t === 'toggle_dark') { darkMode = !darkMode;            renderScreen(); return; }
    }
  }
}

function renderScreen() {
  const screen = document.getElementById('screen');
  screen.innerHTML = '';
  document.body.classList.toggle('dark', darkMode);
  for (const e of getElements()) {
    const div = document.createElement('div');
    div.className = 'el ' + e.role + (e.role === 'toggle' ? (e.selected ? ' on' : ' off') : '');
    div.style.left   = e.bbox[0] + 'px';
    div.style.top    = e.bbox[1] + 'px';
    div.style.width  = (e.bbox[2] - e.bbox[0]) + 'px';
    div.style.height = (e.bbox[3] - e.bbox[1]) + 'px';
    div.textContent  = e.text;
    screen.appendChild(div);
  }
  const ptr = document.getElementById('pointer');
  ptr.style.left = pointerX + 'px';
  ptr.style.top  = pointerY + 'px';
}

window.sendHID = function(vx, vy, c, buttonMask) {
  pointerX = clamp(pointerX + GAIN * vx, 0, W - 1);
  pointerY = clamp(pointerY + GAIN * vy, 0, H - 1);
  lastContact = contact;
  contact = c;
  if (lastContact && !contact) handleClick();
  const ptr = document.getElementById('pointer');
  ptr.style.left = pointerX + 'px';
  ptr.style.top  = pointerY + 'px';
};

window.getMetadata = function() {
  frameCounter++;
  return {
    screen_id:    currentScreen,
    elements:     getElements(),
    pointer_hint: [pointerX, pointerY],
    dark_mode:    darkMode,
    frame_idx:    frameCounter,
  };
};

renderScreen();
</script>
</body>
</html>
"""


class PlaywrightPhone(GUITarget):
    """
    Playwright-driven phone backend using a browser-rendered mock mobile UI.

    Requires optional dependencies: ``pip install playwright Pillow``
    Then: ``python -m playwright install chromium``
    """

    def __init__(
        self,
        key: str,
        width: int = 320,
        height: int = 640,
        *,
        headless: bool = True,
    ) -> None:
        super().__init__(key, width, height)
        self._headless = headless
        self._task_queue: list[str] = ["open settings and toggle dark mode"]
        self._last_ack: Optional[dict] = None
        self._pw_context_manager: Any = None
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._tmpdir: Optional[str] = None
        self._started = False

    # -- lifecycle -----------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._started:
            return

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "PlaywrightPhone requires 'playwright'. "
                "Install with: pip install playwright && python -m playwright install chromium"
            ) from exc

        self._tmpdir = tempfile.mkdtemp(prefix="playwright_phone_")
        html_path = Path(self._tmpdir) / "ios.html"
        html = _MOCK_IOS_HTML.replace("{WIDTH}", str(self.width)).replace("{HEIGHT}", str(self.height))
        html_path.write_text(html)

        self._pw_context_manager = sync_playwright()
        self._playwright = self._pw_context_manager.start()
        self._browser = self._playwright.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page(
            viewport={"width": self.width, "height": self.height},
        )
        self._page.goto(html_path.as_uri())
        self._started = True

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._pw_context_manager.__exit__(None, None, None)
            self._playwright = None
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
        self._started = False

    def __del__(self) -> None:
        self.close()

    # -- GUITarget interface -------------------------------------------------

    def get_new_frame(self) -> Optional[FramePacket]:
        self._ensure_started()
        assert self._page is not None

        screenshot_bytes: bytes = self._page.screenshot()
        tensor = self._png_bytes_to_tensor(screenshot_bytes)

        metadata: dict[str, Any] = self._page.evaluate("window.getMetadata()")
        # Playwright returns lists for bbox; convert to tuples for parity with MockPhone
        for el in metadata.get("elements", []):
            if isinstance(el.get("bbox"), list):
                el["bbox"] = tuple(el["bbox"])
            if isinstance(el.get("pointer_hint"), list):
                el["pointer_hint"] = tuple(el["pointer_hint"])
        if isinstance(metadata.get("pointer_hint"), list):
            metadata["pointer_hint"] = tuple(metadata["pointer_hint"])

        return FramePacket(image=tensor, t_capture_ns=time.time_ns(), metadata=metadata)

    def send_hid(self, vx: float, vy: float, contact: bool, button_mask: int = 1) -> None:
        self._ensure_started()
        assert self._page is not None

        contact_js = "true" if contact else "false"
        self._page.evaluate(f"window.sendHID({vx}, {vy}, {contact_js}, {button_mask})")

        self._last_ack = {
            "vx": vx,
            "vy": vy,
            "contact": contact,
            "button_mask": button_mask,
            "t_ns": time.time_ns(),
        }

    def get_hid_ack(self) -> Optional[dict]:
        ack = self._last_ack
        self._last_ack = None
        return ack

    def get_task_instruction(self) -> Optional[str]:
        if self._task_queue:
            return self._task_queue.pop(0)
        return None

    # -- helpers -------------------------------------------------------------

    def _png_bytes_to_tensor(self, data: bytes) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "PlaywrightPhone requires 'Pillow' for screenshot decoding. "
                "Install with: pip install Pillow"
            ) from exc

        img = Image.open(io.BytesIO(data)).convert("RGB").resize((self.width, self.height))
        # HWC uint8 → CHW float32 [0, 1]
        import numpy as np
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
