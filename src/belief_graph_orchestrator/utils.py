
from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable


def now_ns() -> int:
    return time.time_ns()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def l2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def bbox_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def bbox_area(b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / max(union, 1e-6)


def normalize(vals: list[float]) -> list[float]:
    s = sum(vals)
    if s <= 0:
        return [1.0 / max(len(vals), 1) for _ in vals]
    return [v / s for v in vals]


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None or len(a) != len(b):
        return 0.0
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


def topk(scored: list[tuple[int, float]], k: int) -> list[int]:
    return [nid for nid, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]


def dedupe_preserve_order(xs: Iterable[Any]) -> list[Any]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def sanitize_for_pickle(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: sanitize_for_pickle(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize_for_pickle(v) for v in x]
    return x


def inflate_cov(cov: list[list[float]], q: float = 1.0) -> list[list[float]]:
    return [
        [cov[0][0] + q, cov[0][1], cov[0][2], cov[0][3]],
        [cov[1][0], cov[1][1] + q, cov[1][2], cov[1][3]],
        [cov[2][0], cov[2][1], cov[2][2] + q, cov[2][3]],
        [cov[3][0], cov[3][1], cov[3][2], cov[3][3] + q],
    ]


def shrink_cov(cov: list[list[float]], factor: float = 0.5) -> list[list[float]]:
    return [[v * factor for v in row] for row in cov]
