from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models import BrainModels
from .dataset import TargetSelectionDataset, VerifierDataset
from .losses import pairwise_margin_loss, verifier_loss


class BrainTrainer:
    def __init__(self, device: str = 'cpu', lr: float = 1e-3) -> None:
        self.device = torch.device(device)
        self.models = BrainModels().to(self.device)
        self.optim = torch.optim.Adam(self.models.parameters(), lr=lr)

    def train_verifier(self, dataset: VerifierDataset, batch_size: int = 16, epochs: int = 1) -> None:
        if len(dataset) == 0:
            return
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.models.train()
        for _ in range(epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.models.verifier(x[:, :64])
                loss = verifier_loss(logits, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def train_node_scorer(self, dataset: TargetSelectionDataset, batch_size: int = 16, epochs: int = 1) -> None:
        if len(dataset) == 0:
            return
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.models.train()
        for _ in range(epochs):
            for ctx, pos_vec, pos_extra, neg_vec, neg_extra in loader:
                ctx = ctx[:, :128].to(self.device)
                pos_vec = pos_vec.to(self.device)
                pos_extra = pos_extra.to(self.device)
                neg_vec = neg_vec.to(self.device)
                neg_extra = neg_extra.to(self.device)
                pos_scores = self.models.node_scorer(pos_vec, ctx, pos_extra)
                neg_scores = self.models.node_scorer(neg_vec, ctx, neg_extra)
                loss = pairwise_margin_loss(pos_scores, neg_scores)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.models.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.models.load_state_dict(torch.load(path, map_location=self.device))
