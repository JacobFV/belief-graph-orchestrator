from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..models import BrainModels
from .dataset import TargetSelectionDataset, VerifierDataset


def eval_verifier(models: BrainModels, dataset: VerifierDataset, batch_size: int = 32) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size)
    models.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = models.verifier(x[:, :64])
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return {'acc': correct / max(total, 1), 'n': total}


def eval_node_scorer(models: BrainModels, dataset: TargetSelectionDataset, batch_size: int = 32) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size)
    models.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ctx, pos_vec, pos_extra, neg_vec, neg_extra in loader:
            ctx = ctx[:, :128]
            pos = models.node_scorer(pos_vec, ctx, pos_extra)
            neg = models.node_scorer(neg_vec, ctx, neg_extra)
            correct += int((pos > neg).sum().item())
            total += int(pos.numel())
    return {'pair_acc': correct / max(total, 1), 'n': total}
