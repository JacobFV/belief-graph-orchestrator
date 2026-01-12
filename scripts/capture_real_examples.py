#!/usr/bin/env python3
"""
Run the Brain against real websites via Playwright and capture annotated screenshots.

Usage:
    python scripts/capture_real_examples.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from belief_graph_orchestrator.backends.playwright_desktop import PlaywrightDesktop
from belief_graph_orchestrator.runtime import Brain
from belief_graph_orchestrator.schemas import FramePacket


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGB")


def draw_crosshair(draw: ImageDraw.ImageDraw, x: float, y: float) -> None:
    ix, iy = int(x), int(y)
    arm, w = 14, 2
    draw.line([(ix - arm, iy), (ix + arm, iy)], fill="red", width=w)
    draw.line([(ix, iy - arm), (ix, iy + arm)], fill="red", width=w)
    draw.ellipse([(ix - 5, iy - 5), (ix + 5, iy + 5)], outline="red", width=w)


def draw_element_boxes(draw: ImageDraw.ImageDraw, elements: list[dict]) -> None:
    """Draw semi-transparent bounding boxes around detected elements."""
    colors = {
        "button": "blue",
        "text_field": "green",
        "toggle": "orange",
        "label": "gray",
        "toolbar": "purple",
        "list_item": "cyan",
    }
    for el in elements:
        bbox = el.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        role = el.get("role", "button")
        color = colors.get(role, "blue")
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=1)
        text = (el.get("text") or "")[:25]
        if text:
            draw.text((x1 + 2, y1 + 1), text, fill=color)


def annotate_frame(frame: FramePacket, step: int, summary: dict) -> Image.Image:
    pil = tensor_to_pil(frame.image)
    draw = ImageDraw.Draw(pil)

    # Draw element bounding boxes
    elements = frame.metadata.get("elements", [])
    draw_element_boxes(draw, elements)

    # Draw cursor crosshair
    px, py = frame.metadata.get("pointer_hint", (pil.width // 2, pil.height // 2))
    draw_crosshair(draw, float(px), float(py))

    # Status bar
    bar_h = 20
    draw.rectangle([(0, 0), (pil.width, bar_h)], fill=(0, 0, 0))
    screen = summary.get("screen", "?")[:40]
    subgoal = summary.get("subgoal", "")
    chunk = summary.get("active_chunk", "")
    n_elements = len(elements)
    text = f"step={step}  screen={screen}  sub={subgoal}  chunk={chunk}  els={n_elements}"
    draw.text((4, 3), text, fill="white")

    return pil


def run_real_site(
    name: str,
    url: str,
    task: str,
    steps: int = 80,
    every: int = 4,
) -> None:
    out_dir = ROOT / "examples" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  URL:  {url}")
    print(f"  Task: {task}")
    print(f"{'='*60}")

    desktop = PlaywrightDesktop(
        name,
        width=1280,
        height=800,
        headless=True,
        url=url,
        task=task,
    )
    brain = Brain(
        target_key=name,
        target_instance=desktop,
        use_metadata_hints=True,
    )

    t0 = time.time()
    for i in range(steps):
        brain.step()
        s = brain.summary()

        if i % every == 0:
            elapsed = time.time() - t0
            print(f"  step={i:3d} ({elapsed:.1f}s)  screen={str(s.get('screen',''))[:30]:30s}  "
                  f"sub={str(s.get('subgoal','')):25s}  chunk={str(s.get('active_chunk',''))}")

            frame_events = brain.journal.tail(20, {"frame"})
            if frame_events:
                fp = frame_events[-1].payload.get("frame_packet")
                if fp is not None:
                    img = annotate_frame(fp, i, s)
                    img.save(out_dir / f"step_{i:03d}.png")

    n_saved = len(list(out_dir.glob("*.png")))
    print(f"  -> Saved {n_saved} frames to {out_dir}/")
    desktop.close()


def main():
    sites = [
        {
            "name": "real_wikipedia",
            "url": "https://en.wikipedia.org/wiki/Main_Page",
            "task": "click the search box and type artificial intelligence",
            "steps": 60,
            "every": 3,
        },
        {
            "name": "real_hackernews",
            "url": "https://news.ycombinator.com",
            "task": "click on the first story link",
            "steps": 40,
            "every": 3,
        },
        {
            "name": "real_example_form",
            "url": "https://httpbin.org/forms/post",
            "task": "click the customer name field and type John Smith",
            "steps": 60,
            "every": 3,
        },
    ]

    for site in sites:
        try:
            run_real_site(**site)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
