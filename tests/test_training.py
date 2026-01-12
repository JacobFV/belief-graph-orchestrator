from pathlib import Path
from belief_graph_orchestrator.training.synthetic import generate_mock_sessions
from belief_graph_orchestrator.training.dataset import VerifierDataset, TargetSelectionDataset
from belief_graph_orchestrator.training.trainer import BrainTrainer

def test_training_smoke(tmp_path: Path):
    data_dir = tmp_path / 'data'
    generate_mock_sessions(data_dir, num_sessions=2, steps=20)
    vds = VerifierDataset.from_directory(data_dir)
    if len(vds) > 0:
        trainer = BrainTrainer(device='cpu', lr=1e-3)
        trainer.train_verifier(vds, batch_size=2, epochs=1)
    tds = TargetSelectionDataset.from_directory(data_dir)
    if len(tds) > 0:
        trainer = BrainTrainer(device='cpu', lr=1e-3)
        trainer.train_node_scorer(tds, batch_size=2, epochs=1)
