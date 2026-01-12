#!/usr/bin/env python3
"""
Large-scale real-website testing with full state annotations.

Tests the brain on complex, multi-step tasks across real websites
using both privileged (DOM) and unprivileged (servo/vision-only) modes.
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from belief_graph_orchestrator.runtime import Brain
from belief_graph_orchestrator.schemas import FramePacket


# ── rendering helpers ────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGB")


def draw_crosshair(draw: ImageDraw.ImageDraw, x: float, y: float) -> None:
    ix, iy = int(x), int(y)
    arm, w = 14, 2
    draw.line([(ix - arm, iy), (ix + arm, iy)], fill="red", width=w)
    draw.line([(ix, iy - arm), (ix, iy + arm)], fill="red", width=w)
    draw.ellipse([(ix - 5, iy - 5), (ix + 5, iy + 5)], outline="red", width=w)


def build_state_lines(brain: Brain, step: int, elapsed_s: float) -> list[str]:
    s = brain.state
    lines = []
    lines.append(f"{'─' * 90}")
    lines.append(f"STEP {step}  |  wall={elapsed_s:.2f}s  |  events={len(brain.journal.events)}  |  nodes={len(brain.graph.nodes)}")
    lines.append(f"{'─' * 90}")

    lines.append(f"TASK: {s.task_state.active_goal or '(none)'}")
    sg = s.current_subgoal
    lines.append(f"SUBGOAL: {sg.description if sg else '(none)'}  conf={sg.confidence:.2f}" if sg else "SUBGOAL: (none)")
    lines.append(f"PROGRESS: {s.task_state.progress_estimate:.2f}  RISK: {s.task_state.risk_posture}")
    lines.append("")

    pp = s.pointer
    lines.append(f"POINTER: x={pp.x_hat:.1f}  y={pp.y_hat:.1f}  uncertainty={s.pointer_uncertainty:.4f}  vis={pp.visible_conf:.3f}")

    chunk = s.active_action_chunk
    if chunk:
        elapsed_ms = (s.now_ns - chunk.started_ns) / 1e6
        phase = s.gesture_state.current_phase or "?"
        target = chunk.target_distribution.get("mean", "?")
        lines.append(f"CHUNK: {chunk.kind}  phase={phase}  elapsed={elapsed_ms:.0f}ms/{chunk.timeout_ms}ms  target={target}")
    else:
        lines.append("CHUNK: (none)")

    vs = s.verification_state
    lines.append(f"VERIFY: micro={vs.micro_ok:.2f} servo={vs.servo_ok:.2f} gesture={vs.gesture_ok:.2f} "
                 f"interact={vs.interaction_ok:.2f} subtask={vs.subtask_ok:.2f} task={vs.task_ok:.2f}")

    lines.append(f"METRICS: entropy={s.branch_entropy:.3f} failure={s.failure_density:.3f} "
                 f"fragile={s.fragile_action_phase:.3f} timeout={s.pending_timeout_pressure:.3f}")

    # Scale bands (compact)
    bands = []
    for cid in ["pointer", "task", "verifier", "recovery"]:
        if cid in s.complexes:
            cx = s.complexes[cid]
            top_lvl = max(cx.active_scale_band.level_probs, key=cx.active_scale_band.level_probs.get)
            bands.append(f"{cid}→L{top_lvl}")
    lines.append(f"SCALES: {' | '.join(bands)}")

    # Contracts
    lines.append(f"CONTRACTS: servo_err={s.servo_state.current_error_xy}  "
                 f"subtask_contra={s.subtask_state.contradiction_score:.2f}  "
                 f"interact_ambig={s.interaction_state.ambiguity_score:.2f}")

    kind_counts: dict[str, int] = {}
    for nid in list(brain.graph.nodes.keys()):
        n = brain.graph.node(nid)
        kind_counts[n.kind] = kind_counts.get(n.kind, 0) + 1
    lines.append(f"GRAPH: {len(brain.graph.nodes)} nodes — " + ", ".join(f"{k}={v}" for k, v in sorted(kind_counts.items())))

    if s.live_branch_ids:
        for bid in s.live_branch_ids[:3]:
            b = s.branches.get(bid)
            if b:
                lines.append(f"  branch {bid}: post={b.posterior:.3f} status={b.status}")

    verdicts = brain.journal.tail(30, {"verifier_judgment"})
    if verdicts:
        v = verdicts[-1].payload["verdict"]
        label = v.label if hasattr(v, "label") else v.get("label", "?")
        fs = v.notes.get("failure_scale") if hasattr(v, "notes") else None
        lines.append(f"LAST VERDICT: {label}" + (f"  failure_scale=L{fs}" if fs is not None else ""))

    lines.append(f"SCREEN: {s.latest_screen_id}  prev={s.previous_screen_id}")
    return lines


def render_annotated(frame: FramePacket, brain: Brain, step: int, elapsed_s: float) -> Image.Image:
    screenshot = tensor_to_pil(frame.image)
    draw_ss = ImageDraw.Draw(screenshot)
    px, py = frame.metadata.get("pointer_hint", (screenshot.width // 2, screenshot.height // 2))
    draw_crosshair(draw_ss, float(px), float(py))

    state_lines = build_state_lines(brain, step, elapsed_s)
    line_h = 14
    panel_h = (len(state_lines) + 2) * line_h
    panel = Image.new("RGB", (screenshot.width, panel_h), color=(20, 20, 30))
    draw_p = ImageDraw.Draw(panel)
    for i, line in enumerate(state_lines):
        color = "#ffffff"
        if line.startswith("─"): color = "#555555"
        elif line.startswith("TASK:"): color = "#ffcc00"
        elif line.startswith("CHUNK:"): color = "#00ccff"
        elif line.startswith("POINTER:"): color = "#00ff88"
        elif line.startswith("VERIFY:"): color = "#ff8888"
        elif line.startswith("METRICS:"): color = "#cc88ff"
        elif line.startswith("SCALES:"): color = "#88ffcc"
        elif line.startswith("CONTRACTS:"): color = "#ffaa44"
        elif line.startswith("GRAPH:"): color = "#88ccff"
        elif line.startswith("LAST VERDICT:"):
            color = "#ff4444" if "failure" in line.lower() else "#44ff44"
        draw_p.text((8, line_h * (i + 1)), line, fill=color)

    composite = Image.new("RGB", (screenshot.width, screenshot.height + panel_h))
    composite.paste(screenshot, (0, 0))
    composite.paste(panel, (0, screenshot.height))
    return composite


# ── scenario runner ──────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    name: str
    mode: str
    url: str
    task: str
    steps: int
    wall_s: float
    screens_visited: list[str]
    num_events: int
    num_nodes: int
    num_actions: int
    num_verdicts: int
    success_verdicts: int
    failure_verdicts: int
    frames_saved: int


def run_scenario(
    name: str,
    url: str,
    task: str,
    steps: int = 150,
    every: int = 5,
    mode: str = "dom",  # "dom" or "servo"
) -> ScenarioResult:
    out_dir = ROOT / "examples" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  {name} [{mode.upper()}]")
    print(f"  URL:  {url}")
    print(f"  Task: {task}")
    print(f"  Steps: {steps}")
    print(f"{'=' * 70}")

    if mode == "servo":
        from belief_graph_orchestrator.backends.playwright_servo import PlaywrightServoTarget
        target = PlaywrightServoTarget(name, headless=True, url=url, task=task)
        brain = Brain(target_key=name, target_instance=target, use_metadata_hints=False)
    else:
        from belief_graph_orchestrator.backends.playwright_desktop import PlaywrightDesktop
        target = PlaywrightDesktop(name, headless=True, url=url, task=task)
        brain = Brain(target_key=name, target_instance=target, use_metadata_hints=True)

    screens_seen: list[str] = []
    t0 = time.time()
    frames_saved = 0

    for i in range(steps):
        brain.step()
        s = brain.summary()
        screen = str(s.get("screen", ""))

        if screen and (not screens_seen or screens_seen[-1] != screen):
            screens_seen.append(screen)
            elapsed = time.time() - t0
            print(f"  [{elapsed:6.1f}s] step={i:3d}  NEW SCREEN: {screen[:50]}")

        if i % every == 0:
            elapsed = time.time() - t0
            chunk_str = str(s.get("active_chunk", ""))
            subgoal_str = str(s.get("subgoal", ""))
            print(f"  [{elapsed:6.1f}s] step={i:3d}  screen={screen[:30]:30s}  sub={subgoal_str:25s}  chunk={chunk_str}")

            frame_events = brain.journal.tail(100, {"frame"})
            if frame_events:
                fp = frame_events[-1].payload.get("frame_packet")
                if fp is not None:
                    img = render_annotated(fp, brain, i, elapsed)
                    img.save(out_dir / f"step_{i:03d}.png")
                    frames_saved += 1

    wall_s = time.time() - t0

    # Collect stats
    action_events = [e for e in brain.journal.events if e.type == "action_issued"]
    verdict_events = [e for e in brain.journal.events if e.type == "verifier_judgment"]
    success_count = 0
    failure_count = 0
    for ve in verdict_events:
        v = ve.payload.get("verdict")
        if v:
            label = v.label if hasattr(v, "label") else v.get("label", "")
            if label in {"success", "partial"}:
                success_count += 1
            elif label == "failure":
                failure_count += 1

    result = ScenarioResult(
        name=name, mode=mode, url=url, task=task, steps=steps,
        wall_s=wall_s, screens_visited=screens_seen,
        num_events=len(brain.journal.events), num_nodes=len(brain.graph.nodes),
        num_actions=len(action_events), num_verdicts=len(verdict_events),
        success_verdicts=success_count, failure_verdicts=failure_count,
        frames_saved=frames_saved,
    )

    # Print summary
    print(f"\n  ── RESULTS ──")
    print(f"  Wall time:        {wall_s:.1f}s")
    print(f"  Screens visited:  {screens_seen}")
    print(f"  Events:           {result.num_events}")
    print(f"  Graph nodes:      {result.num_nodes}")
    print(f"  Actions issued:   {result.num_actions}")
    print(f"  Verdicts:         {result.num_verdicts} ({success_count} success, {failure_count} failure)")
    print(f"  Frames saved:     {frames_saved} → {out_dir}/")

    # Save result JSON
    (out_dir / "result.json").write_text(json.dumps({
        "name": result.name, "mode": result.mode, "url": result.url,
        "task": result.task, "steps": result.steps, "wall_s": result.wall_s,
        "screens_visited": result.screens_visited,
        "num_events": result.num_events, "num_nodes": result.num_nodes,
        "num_actions": result.num_actions, "num_verdicts": result.num_verdicts,
        "success_verdicts": result.success_verdicts, "failure_verdicts": result.failure_verdicts,
    }, indent=2))

    target.close()
    return result


def main():
    scenarios = [
        # ── DOM mode (privileged element extraction) ──────────────
        {
            "name": "large_hn_navigate",
            "url": "https://news.ycombinator.com",
            "task": "click on the first story link then go back and click the second story",
            "steps": 120,
            "every": 5,
            "mode": "dom",
        },
        {
            "name": "large_wikipedia_search",
            "url": "https://en.wikipedia.org/wiki/Main_Page",
            "task": "click the search box and type artificial intelligence and click search",
            "steps": 150,
            "every": 5,
            "mode": "dom",
        },
        {
            "name": "large_httpbin_form",
            "url": "https://httpbin.org/forms/post",
            "task": "click the customer name field and type John Smith then click the telephone field and type 5551234",
            "steps": 150,
            "every": 5,
            "mode": "dom",
        },
        # ── Servo mode (vision-only, velocity control) ────────────
        {
            "name": "large_servo_hn",
            "url": "https://news.ycombinator.com",
            "task": "click on the first story link",
            "steps": 200,
            "every": 8,
            "mode": "servo",
        },
        {
            "name": "large_servo_form",
            "url": "https://httpbin.org/forms/post",
            "task": "click the customer name field and type John Smith",
            "steps": 200,
            "every": 8,
            "mode": "servo",
        },
    ]

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        try:
            result = run_scenario(**scenario)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Final summary ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  LARGE-SCALE TEST SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        nav = " → ".join(s[:25] for s in r.screens_visited[:6])
        verdict_str = f"{r.success_verdicts}ok/{r.failure_verdicts}fail"
        print(f"  {r.name:30s} [{r.mode:5s}] {r.wall_s:5.1f}s  actions={r.num_actions:3d}  "
              f"verdicts={verdict_str:12s}  nav={nav}")
    print()


if __name__ == "__main__":
    main()
