# belief-graph-orchestrator

A **multi-scale, event-sourced cognitive control system** for autonomous GUI agents. The brain maintains an append-only evidence journal, a revisable belief graph with typed nodes and counterfactual branches, and multiple budgeted cognitive complexes that traverse scale-space at different frequencies — from 60Hz pointer servo to 1Hz historical reflection.

Platform-agnostic: works on mobile devices (velocity-only HID), desktop (absolute mouse + keyboard), web (Playwright), or any pointer-centric GUI surface. The architecture separates **what happened** (immutable evidence), **what is believed** (revisable graph), **what is loaded** (resident memory), and **what each process can see** (query apertures).

## core commitments

1. **Immutable evidence.** Observations and actions enter an append-only journal. Never mutated.
2. **Revisable beliefs.** Derived entities, affordances, and hypotheses are versioned and supersedable.
3. **Multi-scale cognition.** 8 cognitive complexes with scale bands from L0 (microcontrol) to L5 (task/episode).
4. **Explicit query apertures.** Each complex sees a budgeted, source-stratified slice of the resident graph — not the whole world.
5. **Verifier-first control.** Every action spawns expected-outcome branches. The verifier resolves them.
6. **Hidden-state pointer.** Pointer position is estimated via Kalman filter when not directly observable.
7. **Pretrained perception.** Text encoder (sentence-transformers) and vision encoder (MobileNetV3-Small) provide real semantic understanding. Decision models are trained on agent-specific data.
8. **Streaming I/O.** Text and audio input/output flow through the same event-sourced pipeline.

## quickstart

```bash
pip install -e .
python -m belief_graph_orchestrator demo --steps 120
python -m belief_graph_orchestrator demo --backend mock-desktop --steps 80
python -m belief_graph_orchestrator record-mock --out data/mock_bank --num-sessions 8
python -m belief_graph_orchestrator train-verifier --data-dir data/mock_bank --out checkpoints/verifier.pt
```

## architecture

```
capture -> event journal -> perception -> belief graph -> resident memory
                                                           |
                            scheduler -> query apertures -> complexes
                                                           |
                    pointer servo <- action compiler <- task worker
                         |                                 ^
                    HID velocity                      verifier -> recovery
```

The runtime state is the tuple:

```
S_t = (E_<=t, {G_t^(l)}, R_t, {Q_t^(k,l)}, p_t, B_t, Xi_t)
```

where E is the evidence journal, G the multi-scale belief graph, R the resident subgraph, Q the per-complex query apertures, p the pointer posterior, B the branch set, and Xi the cross-scale contracts.

Scales: L5 task -> L4 subtask -> L3 interaction -> L2 gesture -> L1 servo -> L0 microcontrol.

## repo layout

```
src/belief_graph_orchestrator/
├── runtime.py          # Brain + RuntimeState + tick loop
├── schemas.py          # all typed dataclasses
├── models.py           # pretrained encoders + decision models
├── target.py           # GUITarget protocol + DesktopTarget
├── io_streams.py       # streaming text/audio I/O
├── journal.py          # append-only event journal
├── graph.py            # belief graph store
├── memory.py           # factorized saliency + residency manager
├── perception.py       # edge-based proposals + embedding clustering
├── belief.py           # multi-trunk canonical encoder + graph updates
├── retrieval.py        # scale-conditioned query assembly
├── task.py             # embedding-grounded target selection
├── compiler.py         # action chunks + expectation bundles
├── pointer.py          # Kalman servo + phase catch-up
├── verifier.py         # multi-scale verification (L0-L5)
├── recovery.py         # scale-aware cross-scale recovery
├── reflection.py       # VoR-gated historical reflection
├── scheduler.py        # dynamic scale band adaptation
├── backends/           # MockPhone, MockDesktop, Playwright*, RealPhone
└── training/           # distillation, replay, datasets, losses
tests/                  # 14 tests covering all paths
scripts/                # capture, servo, large-scale test scripts
docs/                   # MATH.md, TRAINING_AND_REPLAY.md, IMPLEMENTATION.md
```

## backends

| Backend | Cursor | Keyboard | Perception | Use case |
|---------|--------|----------|------------|----------|
| `MockPhone` | hidden (velocity) | no | metadata | unit testing |
| `MockDesktop` | direct | yes | metadata | unit testing |
| `PlaywrightPhone` | hidden (velocity) | no | metadata/DOM | browser phone sim |
| `PlaywrightDesktop` | direct | yes | DOM extraction | real websites |
| `PlaywrightServoTarget` | hidden (velocity) | yes | vision-only | hardest test mode |
| `RealPhone` | hidden (velocity) | no | vision-only | **production** |

## connecting a real device

```python
from belief_graph_orchestrator.target import GUITarget
from belief_graph_orchestrator.schemas import FramePacket
from belief_graph_orchestrator.runtime import Brain

class MyDevice(GUITarget):
    def get_new_frame(self):
        img = ...  # screenshot -> torch.Tensor (3, H, W) float32 [0,1]
        return FramePacket(image=img, t_capture_ns=time.time_ns())

    def send_hid(self, vx, vy, contact, button_mask=1):
        ...  # send velocity command over BLE/USB/network

device = MyDevice("my-device")
brain = Brain(target_key="my-device", target_instance=device, use_metadata_hints=False)

while True:
    brain.step()
```

## docs

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — full architectural spec
- [`docs/MATH.md`](./docs/MATH.md) — formal notation, Kalman filter, VoR
- [`docs/TRAINING_AND_REPLAY.md`](./docs/TRAINING_AND_REPLAY.md) — training pipeline
- [`docs/IMPLEMENTATION.md`](./docs/IMPLEMENTATION.md) — module boundaries
