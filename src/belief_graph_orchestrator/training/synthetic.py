from __future__ import annotations

from pathlib import Path

from ..backends.mock import MockPhone
from ..runtime import Brain


def generate_mock_sessions(out_dir: str | Path, num_sessions: int = 4, steps: int = 120) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_sessions):
        brain = Brain(f'mock-{i}', target_cls=MockPhone, use_metadata_hints=True)
        for _ in range(steps):
            brain.step()
        brain.save_bundle(out_dir / f'session_{i:03d}')
