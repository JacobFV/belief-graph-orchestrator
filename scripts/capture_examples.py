#!/usr/bin/env python3
"""
Capture step-by-step screenshots from each backend with cursor crosshair overlay.

Outputs:
  examples/<backend>/step_000.png
  examples/<backend>/step_001.png
  ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from belief_graph_orchestrator.runtime import Brain
from belief_graph_orchestrator.schemas import FramePacket


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """CHW float [0,1] -> PIL RGB."""
    arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGB")


def draw_crosshair(img: Image.Image, x: float, y: float, label: str = "") -> Image.Image:
    """Draw a red crosshair + optional label at (x, y)."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    ix, iy = int(x), int(y)
    arm = 12
    width = 2

    # crosshair lines
    draw.line([(ix - arm, iy), (ix + arm, iy)], fill="red", width=width)
    draw.line([(ix, iy - arm), (ix, iy + arm)], fill="red", width=width)
    # small circle
    r = 4
    draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], outline="red", width=width)

    if label:
        draw.text((ix + arm + 4, iy - 8), label, fill="red")

    return img


def annotate_frame(frame: FramePacket, step: int, screen: str, subgoal: str, chunk: str) -> Image.Image:
    """Convert a FramePacket to an annotated PIL image."""
    pil = tensor_to_pil(frame.image)
    # scale up for readability (2x for phone, 1x for desktop)
    w, h = pil.size
    if w < 640:
        pil = pil.resize((w * 2, h * 2), Image.NEAREST)
        scale = 2
    else:
        scale = 1

    px, py = frame.metadata.get("pointer_hint", (w // 2, h // 2))
    pil = draw_crosshair(pil, float(px) * scale, float(py) * scale)

    draw = ImageDraw.Draw(pil)
    # status bar at top
    bar_h = 22
    draw.rectangle([(0, 0), (pil.width, bar_h)], fill=(0, 0, 0, 180))
    text = f"step={step}  screen={screen}  subgoal={subgoal}  chunk={chunk}"
    draw.text((4, 3), text, fill="white")

    return pil


def run_and_capture(
    brain: Brain,
    out_dir: Path,
    steps: int,
    every: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(steps):
        brain.step()
        if i % every != 0:
            continue
        # get the latest frame event (look back far enough past perception/metric events)
        frame_events = brain.journal.tail(20, {"frame"})
        if not frame_events:
            continue
        fp = frame_events[-1].payload.get("frame_packet")
        if fp is None:
            continue
        s = brain.summary()
        img = annotate_frame(
            fp,
            step=i,
            screen=str(s.get("screen", "?")),
            subgoal=str(s.get("subgoal", "")),
            chunk=str(s.get("active_chunk", "")),
        )
        path = out_dir / f"step_{i:03d}.png"
        img.save(path)
    print(f"  Saved to {out_dir}/  ({len(list(out_dir.glob('*.png')))} frames)")


def main():
    examples = ROOT / "examples"

    # ── Mock Phone ───────────────────────────────────────────────────
    print("Mock Phone...")
    from belief_graph_orchestrator.backends.mock import MockPhone
    brain = Brain(target_key="mock", target_cls=MockPhone, use_metadata_hints=True)
    run_and_capture(brain, examples / "mock_phone", steps=400, every=10)

    # ── Mock Desktop ─────────────────────────────────────────────────
    print("Mock Desktop...")
    from belief_graph_orchestrator.backends.mock_desktop import MockDesktop
    brain = Brain(target_key="mock-desktop", target_instance=MockDesktop("mock-desktop"), use_metadata_hints=True)
    run_and_capture(brain, examples / "mock_desktop", steps=80, every=3)

    # ── Playwright Phone ─────────────────────────────────────────────
    print("Playwright Phone...")
    from belief_graph_orchestrator.backends.playwright import PlaywrightPhone
    phone = PlaywrightPhone("pw-phone", headless=True)
    brain = Brain(target_key="pw-phone", target_instance=phone, use_metadata_hints=True)
    run_and_capture(brain, examples / "playwright_phone", steps=100, every=5)
    phone.close()

    # ── Playwright Desktop ───────────────────────────────────────────
    print("Playwright Desktop...")
    from belief_graph_orchestrator.backends.playwright_desktop import PlaywrightDesktop
    desktop = PlaywrightDesktop("pw-desktop", headless=True)
    brain = Brain(target_key="pw-desktop", target_instance=desktop, use_metadata_hints=True)
    run_and_capture(brain, examples / "playwright_desktop", steps=80, every=3)
    desktop.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
