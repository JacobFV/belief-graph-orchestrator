# implementation notes

this python implementation is intentionally explicit and portable. the module boundaries are chosen so that the stable system contracts can later be ported to swift without changing the conceptual decomposition.

- `schemas.py`: dataclass contracts
- `journal.py`: immutable event stream
- `graph.py`: typed belief graph
- `memory.py`: resident hotness
- `perception.py`: evidence extraction
- `belief.py`: graph revision
- `retrieval.py`: query apertures and workspace packing
- `task.py`: subgoal and target selection
- `compiler.py`: action chunk construction
- `pointer.py`: hidden-state pointer control
- `verifier.py`: branch and verdict update
- `recovery.py`: failure handling
- `reflection.py`: historical retrieval
- `scheduler.py`: multi-scale orchestration
- `training/*`: replay, synthetic data, datasets, losses, training, evaluation

this keeps the reference implementation inspectable and mechanically faithful to the architecture.
