
# belief_graph_orchestrator

A full Python prototype of the multi-scale on-device GUI-agent brain.

## modules

- `target.py` — GUITarget protocol + body adapters
- `journal.py` — append-only event journal
- `graph.py` — multi-scale belief graph
- `memory.py` — resident memory manager
- `models.py` — Torch model stack
- `perception.py` — perception worker
- `belief.py` — belief update engine
- `retrieval.py` — scale-aware query assembly + workspace packing
- `task.py` — task/subtask/interaction worker
- `compiler.py` — action chunk compiler + expectations/branches
- `pointer.py` — pointer posterior + velocity controller
- `verifier.py` — multi-scale verifier
- `recovery.py` — recovery worker
- `reflection.py` — historical reflection worker
- `scheduler.py` — scale-aware scheduler
- `runtime.py` — `Brain` runtime
- `demo.py` — mock end-to-end demo

## quickstart

```python
from belief_graph_orchestrator.runtime import Brain
from belief_graph_orchestrator.backends.mock import MockPhone

brain = Brain("mock-key", target_cls=MockPhone, use_metadata_hints=True)
for _ in range(100):
    brain.step()

print(brain.summary())
```

## swapping in a real device transport

Implement a class compatible with `target.GUITarget`:

```python
class RealPhone(GUITarget):
    def get_new_frame(self) -> Optional[FramePacket]: ...
    def send_hid(self, vx: float, vy: float, contact: bool, button_mask: int = 1) -> None: ...
    def get_hid_ack(self) -> Optional[dict]: ...
    def get_task_instruction(self) -> Optional[str]: ...
```

Then:

```python
brain = Brain("YOUR_KEY", target_cls=RealPhone, use_metadata_hints=False)
```

## notes

The runtime is complete and runnable, but several perception / semantic modules are bootstrap implementations:
- OCR fallback is metadata-backed or stubbed
- region proposals can be metadata-backed or coarse-grid
- semantic deliberation is heuristic-first with trainable Torch modules already wired in
- historical reflection is implemented lightly to avoid blocking the real-time path

This is deliberate: the package is designed so you can replace perception and train the Torch modules without changing the runtime architecture.
