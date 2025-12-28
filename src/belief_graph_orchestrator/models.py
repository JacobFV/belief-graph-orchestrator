
from __future__ import annotations

from typing import Sequence

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from sentence_transformers import SentenceTransformer

EMB_DIM = 128


# ── text encoder: sentence-transformers ──────────────────────────────

class TextEncoder(nn.Module):
    """
    Pretrained sentence-transformers encoder (all-MiniLM-L6-v2).
    22M params, 384-d output, projected to 128-d.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = EMB_DIM) -> None:
        super().__init__()
        self._st = SentenceTransformer(model_name)
        self._st.eval()
        for p in self._st.parameters():
            p.requires_grad = False
        st_dim = self._st.get_sentence_embedding_dimension()
        self.proj = nn.Linear(st_dim, dim)
        self.dim = dim

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.dim)
        with torch.no_grad():
            raw = self._st.encode(list(texts), convert_to_tensor=True, show_progress_bar=False)
        projected = self.proj(raw.float().cpu())
        return F.normalize(projected, dim=-1)


# ── vision encoder: MobileNetV3-Small ────────────────────────────────

class VisionEncoder(nn.Module):
    """
    MobileNetV3-Small pretrained on ImageNet.  2.5M params, 576-d features,
    projected to 128-d.  Backbone frozen, projection layer trainable.
    """

    def __init__(self, dim: int = EMB_DIM) -> None:
        super().__init__()
        backbone = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.proj = nn.Linear(576, dim)
        self.dim = dim
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)
        feat = self.pool(feat).flatten(1)
        return F.normalize(self.proj(feat), dim=-1)


# ── decision models (task-specific, trained not pretrained) ──────────

class NodeScorer(nn.Module):
    def __init__(self, node_dim: int = EMB_DIM, ctx_dim: int = EMB_DIM) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + ctx_dim + 8, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        for p in self.mlp.parameters():
            if p.ndim > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            else:
                nn.init.zeros_(p)

    def forward(self, node_vec: torch.Tensor, ctx_vec: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        x = torch.cat([node_vec, ctx_vec, extra], dim=-1)
        return self.mlp(x).squeeze(-1)


class DeliberatorModel(nn.Module):
    def __init__(self, dim: int = 192, n_heads: int = 4, n_layers: int = 2, num_actions: int = 8) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.action_head = nn.Linear(dim, num_actions)
        self.value_head = nn.Linear(dim, 1)
        self.target_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.action_head.bias)
        nn.init.zeros_(self.value_head.bias)
        nn.init.zeros_(self.target_head.bias)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.encoder(tokens, src_key_padding_mask=mask)
        pooled = h.mean(dim=1)
        return {
            "pooled": pooled,
            "action_logits": self.action_head(pooled),
            "value": self.value_head(pooled).squeeze(-1),
            "target_logits": self.target_head(h).squeeze(-1),
        }


class PointerResidualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 2))
        for p in self.net.parameters():
            if p.ndim > 1:
                nn.init.xavier_uniform_(p, gain=0.05)
            else:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.1 * self.net(x)


class VerifierModel(nn.Module):
    def __init__(self, in_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 5))
        for p in self.net.parameters():
            if p.ndim > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            else:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── top-level container ──────────────────────────────────────────────

class BrainModels(nn.Module):
    """
    All models used by the brain.

    Perception (text, vision): pretrained, frozen backbones with
    trainable projection heads.

    Decision (node scorer, deliberator, pointer residual, verifier):
    initialized for training on agent-specific data.
    """

    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = TextEncoder()
        self.vision_encoder = VisionEncoder()
        self.node_scorer = NodeScorer()
        self.deliberator = DeliberatorModel(dim=192)
        self.pointer_residual = PointerResidualModel()
        self.verifier = VerifierModel()

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        return self.text_encoder(texts)

    @torch.no_grad()
    def encode_crops(self, crops: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(crops)

    @torch.no_grad()
    def score_nodes(self, node_vecs: torch.Tensor, ctx_vec: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        ctx = ctx_vec.unsqueeze(0).expand(node_vecs.shape[0], -1)
        return self.node_scorer(node_vecs, ctx, extra)

    @torch.no_grad()
    def deliberate(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.deliberator(tokens, mask)

    @torch.no_grad()
    def pointer_resid(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointer_residual(x)

    @torch.no_grad()
    def verify_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.verifier(x)
