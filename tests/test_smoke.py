from belief_graph_orchestrator.runtime import Brain
from belief_graph_orchestrator.backends import MockPhone

def test_brain_smoke():
    brain = Brain('mock', target_cls=MockPhone, use_metadata_hints=True)
    for _ in range(20):
        brain.step()
    s = brain.summary()
    assert s['num_events'] > 0
    assert s['num_nodes'] > 0
