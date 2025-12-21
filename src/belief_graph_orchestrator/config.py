from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Type, TypeVar

@dataclass
class RuntimeConfig:
    use_metadata_hints: bool = True
    demo_steps: int = 120

@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 1
    seed: int = 7
    device: str = 'cpu'

T = TypeVar('T')

def load_json_config(path: str | Path, cls: Type[T]) -> T:
    with open(path, 'r') as f:
        data = json.load(f)
    return cls(**data)

def save_json_config(obj: Any, path: str | Path) -> None:
    with open(path, 'w') as f:
        json.dump(asdict(obj), f, indent=2)
