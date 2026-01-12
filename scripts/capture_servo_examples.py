#!/usr/bin/env python3
"""
Run the Brain with servo-controlled cursor on real websites.
Vision-only perception (no DOM element extraction).
Screenshots annotated with internal state BELOW the image (agent never sees annotations).

Usage:
    python scripts/capture_servo_examples.py
"""
from __future__ import annotations

import sys
import time
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from belief_graph_orchestrator.backends.playwright_servo import PlaywrightServoTarget
from belief_graph_orchestrator.runtime import Brain
from belief_graph_orchestrator.schemas import FramePacket


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGB")


def build_state_panel(brain: Brain, step: int, elapsed_s: float) -> list[str]:
    """Build lines of internal state text for the annotation panel."""
    s = brain.state
    lines = []
    lines.append(f"{'─'*80}")
    lines.append(f"STEP {step}  |  wall={elapsed_s:.2f}s  |  events={len(brain.journal.events)}  |  nodes={len(brain.graph.nodes)}")
    lines.append(f"{'─'*80}")

    # Task
    lines.append(f"TASK: {s.task_state.active_goal or '(none)'}")
    sg = s.current_subgoal
    lines.append(f"SUBGOAL: {sg.description if sg else '(none)'}  conf={sg.confidence:.2f}" if sg else "SUBGOAL: (none)")
    lines.append(f"PROGRESS: {s.task_state.progress_estimate:.2f}")
    lines.append("")

    # Pointer
    pp = s.pointer
    lines.append(f"POINTER: x={pp.x_hat:.1f}  y={pp.y_hat:.1f}  vx={pp.vx_hat:.2f}  vy={pp.vy_hat:.2f}")
    lines.append(f"  uncertainty={s.pointer_uncertainty:.4f}  visible_conf={pp.visible_conf:.3f}")
    lines.append(f"  cov_diag=[{pp.cov[0][0]:.2f}, {pp.cov[1][1]:.2f}, {pp.cov[2][2]:.2f}, {pp.cov[3][3]:.2f}]")
    lines.append("")

    # Active chunk
    chunk = s.active_action_chunk
    if chunk:
        elapsed_ms = (s.now_ns - chunk.started_ns) / 1e6
        phase = s.gesture_state.current_phase or "?"
        target = chunk.target_distribution.get("mean", "?")
        lines.append(f"CHUNK: {chunk.kind}  id={chunk.id[:20]}  phase={phase}")
        lines.append(f"  elapsed={elapsed_ms:.0f}ms / {chunk.timeout_ms}ms  target={target}")
        lines.append(f"  phases: {' → '.join(p.name for p in chunk.phases)}")
    else:
        lines.append("CHUNK: (none)")
    lines.append("")

    # Expectations + branches
    if s.live_branch_ids:
        lines.append(f"BRANCHES: {len(s.live_branch_ids)} live")
        for bid in s.live_branch_ids[:4]:
            b = s.branches.get(bid)
            if b:
                lines.append(f"  branch {bid}: posterior={b.posterior:.3f}  status={b.status}  exps={b.expectation_ids}")
    else:
        lines.append("BRANCHES: (none)")
    lines.append("")

    # Scalar metrics
    lines.append(f"METRICS: branch_entropy={s.branch_entropy:.3f}  failure_density={s.failure_density:.3f}")
    lines.append(f"  fragile_phase={s.fragile_action_phase:.3f}  timeout_pressure={s.pending_timeout_pressure:.3f}")
    lines.append(f"  ambiguity={s.ambiguity_score:.3f}  analogy_match={s.analogy_match_score:.3f}")
    lines.append("")

    # Belief graph summary
    kind_counts: dict[str, int] = {}
    for nid in list(brain.graph.nodes.keys()):
        n = brain.graph.node(nid)
        kind_counts[n.kind] = kind_counts.get(n.kind, 0) + 1
    lines.append(f"GRAPH: {len(brain.graph.nodes)} nodes — " + ", ".join(f"{k}={v}" for k, v in sorted(kind_counts.items())))

    # Recent verifier judgments
    verdicts = brain.journal.tail(20, {"verifier_judgment"})
    if verdicts:
        v = verdicts[-1].payload["verdict"]
        label = v.label if hasattr(v, "label") else v.get("label", "?")
        lines.append(f"LAST VERDICT: {label}")
    lines.append("")

    # Recovery
    if s.recovery_reasons:
        lines.append(f"RECOVERY: {s.recovery_reasons}")
    lines.append(f"SCREEN: {s.latest_screen_id}  prev={s.previous_screen_id}")

    return lines


def render_annotated_frame(
    frame: FramePacket,
    brain: Brain,
    step: int,
    elapsed_s: float,
) -> Image.Image:
    """
    Build a composite image:
      - Top: the screenshot (exactly what the agent sees)
      - Bottom: state annotation panel (the agent NEVER sees this)

    The FramePacket.image tensor is not modified.
    """
    screenshot = tensor_to_pil(frame.image)
    sw, sh = screenshot.size

    # Draw cursor crosshair on screenshot copy (for human viewing only)
    draw_ss = ImageDraw.Draw(screenshot)
    px, py = frame.metadata.get("pointer_hint", (sw // 2, sh // 2))
    px, py = int(px), int(py)
    arm, w = 14, 2
    draw_ss.line([(px - arm, py), (px + arm, py)], fill="red", width=w)
    draw_ss.line([(px, py - arm), (px, py + arm)], fill="red", width=w)
    draw_ss.ellipse([(px - 5, py - 5), (px + 5, py + 5)], outline="red", width=w)

    # Build state panel text
    state_lines = build_state_panel(brain, step, elapsed_s)

    # Render panel
    line_h = 14
    panel_h = (len(state_lines) + 2) * line_h
    panel = Image.new("RGB", (sw, panel_h), color=(20, 20, 30))
    draw_p = ImageDraw.Draw(panel)
    for i, line in enumerate(state_lines):
        color = "#ffffff"
        if line.startswith("─"):
            color = "#666666"
        elif line.startswith("TASK:"):
            color = "#ffcc00"
        elif line.startswith("CHUNK:"):
            color = "#00ccff"
        elif line.startswith("POINTER:"):
            color = "#00ff88"
        elif line.startswith("BRANCHES:"):
            color = "#ff8800"
        elif line.startswith("METRICS:"):
            color = "#cc88ff"
        elif line.startswith("GRAPH:"):
            color = "#88ccff"
        elif line.startswith("LAST VERDICT:"):
            color = "#ff4444" if "failure" in line.lower() else "#44ff44"
        draw_p.text((8, line_h * (i + 1)), line, fill=color)

    # Composite: screenshot on top, panel below
    composite = Image.new("RGB", (sw, sh + panel_h), color=(0, 0, 0))
    composite.paste(screenshot, (0, 0))
    composite.paste(panel, (0, sh))
    return composite


def run_scenario(
    name: str,
    url: str,
    task: str,
    steps: int = 100,
    every: int = 5,
) -> None:
    out_dir = ROOT / "examples" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  URL:  {url}")
    print(f"  Task: {task}")
    print(f"  Mode: SERVO (velocity control, vision-only perception)")
    print(f"{'='*70}")

    target = PlaywrightServoTarget(
        name,
        width=1280,
        height=800,
        headless=True,
        url=url,
        task=task,
    )

    brain = Brain(
        target_key=name,
        target_instance=target,
        use_metadata_hints=False,  # Vision only — no DOM element extraction
    )

    t0 = time.time()
    for i in range(steps):
        brain.step()
        s = brain.summary()

        if i % every == 0:
            elapsed = time.time() - t0

            # Get latest frame
            frame_events = brain.journal.tail(20, {"frame"})
            if frame_events:
                fp = frame_events[-1].payload.get("frame_packet")
                if fp is not None:
                    img = render_annotated_frame(fp, brain, i, elapsed)
                    img.save(out_dir / f"step_{i:03d}.png")

            print(f"  step={i:3d} ({elapsed:.1f}s)  screen={str(s.get('screen',''))[:30]:30s}  "
                  f"sub={str(s.get('subgoal','')):25s}  chunk={str(s.get('active_chunk',''))}")

    n_saved = len(list(out_dir.glob("*.png")))
    print(f"  -> Saved {n_saved} frames to {out_dir}/")
    target.close()


def main():
    scenarios = [
        {
            "name": "servo_hackernews",
            "url": "https://news.ycombinator.com",
            "task": "click on the first story link",
            "steps": 80,
            "every": 4,
        },
        {
            "name": "servo_wikipedia",
            "url": "https://en.wikipedia.org/wiki/Main_Page",
            "task": "click the search box and type artificial intelligence",
            "steps": 100,
            "every": 5,
        },
        {
            "name": "servo_httpbin_form",
            "url": "https://httpbin.org/forms/post",
            "task": "click the customer name field and type John Smith",
            "steps": 100,
            "every": 5,
        },
    ]

    for scenario in scenarios:
        try:
            run_scenario(**scenario)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
