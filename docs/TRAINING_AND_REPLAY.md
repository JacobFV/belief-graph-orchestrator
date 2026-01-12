# training and replay

this repo includes a real replay/training stack because the runtime already produces the right artifacts.

## session bundles

a session bundle stores:
- summary,
- raw events,
- graph nodes,
- graph edges,
- expectations,
- branches.

it is serialized as `bundle.pkl` plus human-readable summaries.

## replay

`ReplayTarget` replays:
- task instructions,
- HID acks,
- frame packets,

into a fresh `Brain` instance. the point is structural replay through the actual runtime, not idealized offline evaluation.

## synthetic traces

`generate_mock_sessions(...)` uses the mock phone and the real runtime to create bootstrap traces that already contain:
- action chunks,
- expectation bundles,
- verifier judgments,
- graph revisions,
- metric snapshots.

## datasets

### verifier dataset

for every verifier judgment, the nearest preceding metric snapshot is turned into a feature vector. label is the verifier verdict.

### target selection dataset

for each root action emission, the selected target node becomes a positive node and a random other node becomes a negative. this supports pairwise node-scorer training.

## trainer

`BrainTrainer` currently exposes:
- `train_verifier(...)`
- `train_node_scorer(...)`

and saves checkpoints with `torch.save`.
