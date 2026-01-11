
from __future__ import annotations

import io
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch

from ..target import DesktopTarget
from ..schemas import FramePacket

# ── JS snippet: extract interactive elements from any real DOM ───────
_EXTRACT_ELEMENTS_JS = """
() => {
  const out = [];
  const seen = new Set();
  const sel = 'a, button, input, textarea, select, [role="button"], ' +
              '[role="link"], [role="menuitem"], [role="tab"], [role="checkbox"], ' +
              '[role="switch"], [role="textbox"], [role="searchbox"], ' +
              '[onclick], [tabindex], label, h1, h2, h3, h4, nav, header, footer';
  document.querySelectorAll(sel).forEach((el, i) => {
    const rect = el.getBoundingClientRect();
    if (rect.width < 4 || rect.height < 4) return;
    if (rect.bottom < 0 || rect.top > window.innerHeight) return;
    if (rect.right < 0 || rect.left > window.innerWidth) return;
    const key = Math.round(rect.left) + ',' + Math.round(rect.top) + ',' +
                Math.round(rect.width) + ',' + Math.round(rect.height);
    if (seen.has(key)) return;
    seen.add(key);
    const tag = el.tagName.toLowerCase();
    const ariaRole = el.getAttribute('role') || '';
    const text = (el.textContent || '').trim().substring(0, 80) ||
                 el.value || el.placeholder ||
                 el.getAttribute('aria-label') || el.title || '';
    if (!text && tag !== 'input' && tag !== 'textarea') return;
    let role = 'button';
    if (tag === 'input' || tag === 'textarea' || ariaRole === 'textbox' || ariaRole === 'searchbox')
      role = 'text_field';
    else if (tag === 'select') role = 'button';
    else if (tag === 'a') role = 'button';
    else if (tag === 'nav' || tag === 'header' || tag === 'footer') role = 'toolbar';
    else if (/^h[1-4]$/.test(tag)) role = 'label';
    else if (tag === 'label') role = 'label';
    else if (ariaRole === 'checkbox' || ariaRole === 'switch') role = 'toggle';
    out.push({
      id:       'dom_' + i,
      text:     text,
      bbox:     [rect.left, rect.top, rect.right, rect.bottom],
      role:     role,
      action:   {},
      selected: el.checked || false,
      enabled:  !el.disabled,
    });
  });
  return out;
}
"""

# ── Mock desktop HTML (unchanged from before) ────────────────────────
_MOCK_DESKTOP_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    width: {WIDTH}px; height: {HEIGHT}px; overflow: hidden;
    font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
    background: #f0f0f0;
  }
  #screen { width: 100%; height: 100%; position: relative; }
  .el {
    position: absolute;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 500; color: #fff;
    border-radius: 4px; overflow: hidden; white-space: nowrap;
  }
  .el.label      { color: #222; background: none; font-weight: 600; font-size: 15px; }
  .el.button     { background: #3d78d8; cursor: pointer; }
  .el.button:hover { background: #2c5fb3; }
  .el.toggle.on  { background: #34a853; }
  .el.toggle.off { background: #b33; }
  .el.text_field { background: #fff; color: #333; border: 1px solid #bbb; border-radius: 3px; }
  .el.list_item  { background: #e6e8ee; color: #222; border-bottom: 1px solid #ccc; border-radius: 0; }
  .el.menubar    { background: #383838; color: #ddd; font-size: 12px; border-radius: 0; }
  .el.sidebar    { background: #e0e0e3; color: #333; border-radius: 0; }
  #cursor {
    position: absolute; width: 14px; height: 14px;
    z-index: 9999; pointer-events: none;
    transform: translate(-50%, -50%);
  }
  #cursor::before, #cursor::after {
    content: ''; position: absolute; background: red;
  }
  #cursor::before { width: 14px; height: 2px; top: 6px; left: 0; }
  #cursor::after  { width: 2px; height: 14px; top: 0; left: 6px; }
</style>
</head>
<body>
<div id="screen"></div>
<div id="cursor"></div>
<script>
const W = {WIDTH}, H = {HEIGHT};

let currentScreen = 'main';
let textBuffer = '';
let checkboxChecked = false;
let sidebarOpen = true;
let cursorX = W / 2;
let cursorY = H / 2;
let frameCounter = 0;

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function inside(x, y, b) { return b[0] <= x && x <= b[2] && b[1] <= y && y <= b[3]; }

function getElements() {
  const elems = [];
  if (currentScreen === 'main') {
    elems.push({id:'menubar', text:'File  Edit  View  Help', bbox:[0,0,W,30], role:'menubar', action:{}, selected:false, enabled:true});
    elems.push({id:'title', text:'Desktop App', bbox:[W/2-80,40,W/2+80,70], role:'label', action:{}, selected:false, enabled:true});
    if (sidebarOpen) {
      elems.push({id:'sidebar_bg', text:'sidebar', bbox:[0,30,220,H], role:'sidebar', action:{}, selected:false, enabled:true});
      elems.push({id:'nav_home', text:'Home', bbox:[10,80,210,120], role:'button', action:{type:'goto',screen:'main'}, selected:false, enabled:true});
      elems.push({id:'nav_form', text:'Form', bbox:[10,130,210,170], role:'button', action:{type:'goto',screen:'form'}, selected:false, enabled:true});
      elems.push({id:'nav_table', text:'Data Table', bbox:[10,180,210,220], role:'button', action:{type:'goto',screen:'table'}, selected:false, enabled:true});
    }
    const cl = sidebarOpen ? 230 : 10;
    elems.push({id:'welcome', text:'Welcome to the desktop app.', bbox:[cl,100,W-20,140], role:'label', action:{}, selected:false, enabled:true});
    elems.push({id:'submit', text:'Submit', bbox:[cl,200,cl+160,250], role:'button', action:{type:'goto',screen:'form'}, selected:false, enabled:true});
    elems.push({id:'toggle_sidebar', text:'Toggle Sidebar', bbox:[cl,280,cl+180,320], role:'button', action:{type:'toggle_sidebar'}, selected:false, enabled:true});
  } else if (currentScreen === 'form') {
    elems.push({id:'menubar', text:'File  Edit  View  Help', bbox:[0,0,W,30], role:'menubar', action:{}, selected:false, enabled:true});
    elems.push({id:'back', text:'Back', bbox:[10,40,80,70], role:'button', action:{type:'goto',screen:'main'}, selected:false, enabled:true});
    elems.push({id:'title', text:'Form', bbox:[W/2-40,40,W/2+40,70], role:'label', action:{}, selected:false, enabled:true});
    elems.push({id:'name_label', text:'Name:', bbox:[40,120,140,150], role:'label', action:{}, selected:false, enabled:true});
    elems.push({id:'name_field', text:textBuffer||'Enter name...', bbox:[150,110,W-40,155], role:'text_field', action:{type:'focus'}, selected:false, enabled:true});
    elems.push({id:'checkbox', text:'Accept terms', bbox:[40,200,260,240], role:'toggle', action:{type:'toggle_check'}, selected:checkboxChecked, enabled:true});
    elems.push({id:'submit_btn', text:'Submit', bbox:[40,300,200,350], role:'button', action:{type:'submit'}, selected:false, enabled:checkboxChecked});
    elems.push({id:'cancel_btn', text:'Cancel', bbox:[220,300,380,350], role:'button', action:{type:'goto',screen:'main'}, selected:false, enabled:true});
  } else if (currentScreen === 'table') {
    elems.push({id:'menubar', text:'File  Edit  View  Help', bbox:[0,0,W,30], role:'menubar', action:{}, selected:false, enabled:true});
    elems.push({id:'back', text:'Back', bbox:[10,40,80,70], role:'button', action:{type:'goto',screen:'main'}, selected:false, enabled:true});
    elems.push({id:'title', text:'Data Table', bbox:[W/2-60,40,W/2+60,70], role:'label', action:{}, selected:false, enabled:true});
    elems.push({id:'header', text:'ID  |  Name  |  Status', bbox:[40,100,W-40,135], role:'label', action:{}, selected:false, enabled:true});
    for (let i = 0; i < 6; i++) {
      const y0 = 145 + i * 40;
      elems.push({id:'row_'+i, text:(i+1)+'  |  Item '+(i+1)+'  |  Active', bbox:[40,y0,W-40,y0+35], role:'list_item', action:{}, selected:false, enabled:true});
    }
  }
  return elems;
}

function handleClick() {
  for (const e of getElements()) {
    if (inside(cursorX, cursorY, e.bbox) && e.enabled) {
      const t = (e.action||{}).type;
      if (t === 'goto')           { currentScreen = e.action.screen; renderScreen(); return; }
      if (t === 'toggle_sidebar') { sidebarOpen = !sidebarOpen;      renderScreen(); return; }
      if (t === 'toggle_check')   { checkboxChecked = !checkboxChecked; renderScreen(); return; }
      if (t === 'submit')         { currentScreen = 'main'; textBuffer = ''; checkboxChecked = false; renderScreen(); return; }
    }
  }
}

function renderScreen() {
  const screen = document.getElementById('screen');
  screen.innerHTML = '';
  for (const e of getElements()) {
    const div = document.createElement('div');
    let cls = 'el ' + e.role;
    if (e.role === 'toggle') cls += e.selected ? ' on' : ' off';
    if (!e.enabled) div.style.opacity = '0.4';
    div.className = cls;
    div.style.left   = e.bbox[0] + 'px';
    div.style.top    = e.bbox[1] + 'px';
    div.style.width  = (e.bbox[2] - e.bbox[0]) + 'px';
    div.style.height = (e.bbox[3] - e.bbox[1]) + 'px';
    div.textContent  = e.text;
    screen.appendChild(div);
  }
  updateCursor();
}

function updateCursor() {
  const c = document.getElementById('cursor');
  c.style.left = cursorX + 'px';
  c.style.top  = cursorY + 'px';
}

window.moveCursorTo = function(x, y) {
  cursorX = clamp(x, 0, W - 1);
  cursorY = clamp(y, 0, H - 1);
  updateCursor();
};

window.doClick = function(x, y) {
  cursorX = clamp(x, 0, W - 1);
  cursorY = clamp(y, 0, H - 1);
  handleClick();
  updateCursor();
};

window.doKey = function(key, modifiers) {
  if (key === 'Escape' && currentScreen !== 'main') { currentScreen = 'main'; renderScreen(); }
  if (key === 'ArrowLeft' && (modifiers||[]).includes('Alt') && currentScreen !== 'main') { currentScreen = 'main'; renderScreen(); }
};

window.doText = function(text) {
  textBuffer += text;
  renderScreen();
};

window.getCursorPos = function() { return [cursorX, cursorY]; };

window.getMetadata = function() {
  frameCounter++;
  return {
    screen_id:    currentScreen,
    elements:     getElements(),
    pointer_hint: [cursorX, cursorY],
    frame_idx:    frameCounter,
  };
};

renderScreen();
</script>
</body>
</html>
"""


class PlaywrightDesktop(DesktopTarget):
    """
    Playwright-driven desktop backend.

    Two modes:
      - **Mock mode** (url=None): opens a built-in mock desktop HTML app.
      - **Real mode** (url="https://..."): opens a real webpage, extracts
        interactive elements from the DOM via accessibility queries, and
        drives it with real Playwright mouse/keyboard events.

    Requires: ``pip install playwright Pillow``
    Then:     ``python -m playwright install chromium``
    """

    def __init__(
        self,
        key: str,
        width: int = 1280,
        height: int = 800,
        *,
        headless: bool = True,
        url: str | None = None,
        task: str | None = None,
    ) -> None:
        super().__init__(key, width, height)
        self._headless = headless
        self._url = url  # None → mock HTML, otherwise real page
        self._task_queue: list[str] = [task] if task else (
            [] if url else ["click the submit button and type hello"]
        )
        self._last_ack: Optional[dict] = None
        self._cursor_x: float = width / 2.0
        self._cursor_y: float = height / 2.0
        self._pw_context_manager: Any = None
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._tmpdir: Optional[str] = None
        self._started = False
        self._is_real_page: bool = url is not None
        self._frame_counter = 0

    # ── lifecycle ────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        if self._started:
            return

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "PlaywrightDesktop requires 'playwright'. "
                "Install with: pip install playwright && python -m playwright install chromium"
            ) from exc

        self._pw_context_manager = sync_playwright()
        self._playwright = self._pw_context_manager.start()
        self._browser = self._playwright.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page(viewport={"width": self.width, "height": self.height})

        if self._url:
            self._page.goto(self._url, wait_until="domcontentloaded", timeout=15000)
            # Give the page a moment to settle
            self._page.wait_for_timeout(500)
        else:
            self._tmpdir = tempfile.mkdtemp(prefix="pw_desktop_")
            html_path = Path(self._tmpdir) / "desktop.html"
            html = _MOCK_DESKTOP_HTML.replace("{WIDTH}", str(self.width)).replace("{HEIGHT}", str(self.height))
            html_path.write_text(html)
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

    # ── GUITarget interface ──────────────────────────────────────────

    def get_new_frame(self) -> Optional[FramePacket]:
        self._ensure_started()
        assert self._page is not None
        self._frame_counter += 1

        try:
            screenshot_bytes: bytes = self._page.screenshot()
        except Exception:
            # Page may be navigating or loading — return None this tick
            return None
        tensor = self._png_bytes_to_tensor(screenshot_bytes)

        if self._is_real_page:
            metadata = self._extract_real_page_metadata()
        else:
            metadata = self._page.evaluate("window.getMetadata()")
            for el in metadata.get("elements", []):
                if isinstance(el.get("bbox"), list):
                    el["bbox"] = tuple(el["bbox"])
            if isinstance(metadata.get("pointer_hint"), list):
                metadata["pointer_hint"] = tuple(metadata["pointer_hint"])

        return FramePacket(image=tensor, t_capture_ns=time.time_ns(), metadata=metadata)

    def move_cursor_to(self, x: float, y: float) -> None:
        self._ensure_started()
        assert self._page is not None
        self._cursor_x, self._cursor_y = x, y
        if self._is_real_page:
            self._page.mouse.move(x, y)
        else:
            self._page.evaluate(f"window.moveCursorTo({x}, {y})")
        self._last_ack = {"type": "move", "x": x, "y": y, "t_ns": time.time_ns()}

    def click(self, x: float, y: float, button: str = "left") -> None:
        self._ensure_started()
        assert self._page is not None
        self._cursor_x, self._cursor_y = x, y
        if self._is_real_page:
            self._page.mouse.click(x, y, button=button)
        else:
            self._page.evaluate(f"window.doClick({x}, {y})")
        self._last_ack = {"type": "click", "x": x, "y": y, "button": button, "t_ns": time.time_ns()}

    def send_key(self, key: str, modifiers: list[str] | None = None) -> None:
        self._ensure_started()
        assert self._page is not None
        if self._is_real_page:
            combo = "+".join((modifiers or []) + [key])
            self._page.keyboard.press(combo)
        else:
            import json as _json
            mods_js = _json.dumps(modifiers or [])
            self._page.evaluate(f"window.doKey({_json.dumps(key)}, {mods_js})")
        self._last_ack = {"type": "key", "key": key, "modifiers": modifiers or [], "t_ns": time.time_ns()}

    def send_text(self, text: str) -> None:
        self._ensure_started()
        assert self._page is not None
        if self._is_real_page:
            self._page.keyboard.type(text, delay=20)
        else:
            import json as _json
            self._page.evaluate(f"window.doText({_json.dumps(text)})")
        self._last_ack = {"type": "text", "text": text, "t_ns": time.time_ns()}

    def get_cursor_position(self) -> Optional[tuple[float, float]]:
        if self._is_real_page or not self._started:
            return (self._cursor_x, self._cursor_y)
        assert self._page is not None
        pos = self._page.evaluate("window.getCursorPos()")
        self._cursor_x, self._cursor_y = float(pos[0]), float(pos[1])
        return (self._cursor_x, self._cursor_y)

    def get_hid_ack(self) -> Optional[dict]:
        ack = self._last_ack
        self._last_ack = None
        return ack

    def get_task_instruction(self) -> Optional[str]:
        if self._task_queue:
            return self._task_queue.pop(0)
        return None

    # ── real page metadata extraction ────────────────────────────────

    def _extract_real_page_metadata(self) -> dict[str, Any]:
        assert self._page is not None
        try:
            elements: list[dict] = self._page.evaluate(_EXTRACT_ELEMENTS_JS)
        except Exception:
            elements = []
        # Normalize bbox arrays → tuples
        for el in elements:
            if isinstance(el.get("bbox"), list):
                el["bbox"] = tuple(el["bbox"])
        title = self._page.title() or ""
        url = self._page.url or ""
        return {
            "screen_id": title[:40] or url[:40],
            "elements": elements,
            "pointer_hint": (self._cursor_x, self._cursor_y),
            "frame_idx": self._frame_counter,
            "page_title": title,
            "page_url": url,
        }

    # ── helpers ──────────────────────────────────────────────────────

    def _png_bytes_to_tensor(self, data: bytes) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "PlaywrightDesktop requires 'Pillow' for screenshot decoding. "
                "Install with: pip install Pillow"
            ) from exc

        import numpy as np
        img = Image.open(io.BytesIO(data)).convert("RGB").resize((self.width, self.height))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
