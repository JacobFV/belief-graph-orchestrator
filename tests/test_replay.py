from pathlib import Path
from belief_graph_orchestrator.runtime import Brain
from belief_graph_orchestrator.backends import MockPhone
from belief_graph_orchestrator.serialization import load_session_bundle
from belief_graph_orchestrator.training.replay import replay_bundle

def test_replay_roundtrip(tmp_path: Path):
    brain = Brain('mock', target_cls=MockPhone, use_metadata_hints=True)
    for _ in range(20):
        brain.step()
    out = tmp_path / 'bundle'
    brain.save_bundle(out)
    bundle = load_session_bundle(out)
    summary = replay_bundle(bundle, steps=20)
    assert 'num_events' in summary
