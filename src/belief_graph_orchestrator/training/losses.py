from __future__ import annotations

import torch
import torch.nn.functional as F


def verifier_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def pairwise_margin_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    return torch.relu(margin - pos_scores + neg_scores).mean()


# ── pseudo-AX structure distillation losses ──────────────────────────
#
# The spec says: "train a model to infer pseudo-AX latent graph from
# screenshots.  Use privileged structure at training time to produce an
# unprivileged inference model at runtime."
#
# These losses supervise the perception stack to predict DOM/AX-like
# structure from pixels, using privileged teacher data collected from
# simulators, web DOMs, Android accessibility trees, etc.

def role_distillation_loss(
    predicted_role_logits: torch.Tensor,
    teacher_role_labels: torch.Tensor,
) -> torch.Tensor:
    """Supervise affordance role prediction from privileged teacher labels."""
    return F.cross_entropy(predicted_role_logits, teacher_role_labels)


def containment_distillation_loss(
    predicted_containment: torch.Tensor,
    teacher_containment: torch.Tensor,
) -> torch.Tensor:
    """Supervise container/parent-child structure prediction."""
    return F.binary_cross_entropy_with_logits(predicted_containment, teacher_containment)


def same_entity_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Contrastive loss for same-entity tracking across time.
    L_same = -log( exp(s(i,j+)/τ) / (exp(s(i,j+)/τ) + Σ exp(s(i,j-)/τ)) )
    """
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / temperature
    neg_sim = F.cosine_similarity(
        anchor.unsqueeze(1), negatives, dim=-1
    ) / temperature  # (B, N)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+N)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def pointer_observation_loss(
    predicted_xy: torch.Tensor,
    true_xy: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for pointer observation matching (dev-time true pointer labels)."""
    return F.mse_loss(predicted_xy, true_xy)


def branch_prediction_loss(
    predicted_priors: torch.Tensor,
    actual_outcomes: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between predicted branch priors and actual outcomes."""
    predicted_log = F.log_softmax(predicted_priors, dim=-1)
    actual_dist = F.softmax(actual_outcomes, dim=-1)
    return F.kl_div(predicted_log, actual_dist, reduction="batchmean")


def full_distillation_loss(
    role_logits: torch.Tensor | None = None,
    role_labels: torch.Tensor | None = None,
    containment_pred: torch.Tensor | None = None,
    containment_true: torch.Tensor | None = None,
    entity_anchor: torch.Tensor | None = None,
    entity_pos: torch.Tensor | None = None,
    entity_neg: torch.Tensor | None = None,
    pointer_pred: torch.Tensor | None = None,
    pointer_true: torch.Tensor | None = None,
    branch_pred: torch.Tensor | None = None,
    branch_true: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Combined distillation loss:
      L = λ₁L_role + λ₂L_contain + λ₃L_same + λ₄L_pointer + λ₅L_branch
    """
    w = weights or {
        "role": 1.0,
        "contain": 0.5,
        "same_entity": 1.0,
        "pointer": 0.5,
        "branch": 0.8,
    }
    total = torch.tensor(0.0)
    if role_logits is not None and role_labels is not None:
        total = total + w.get("role", 1.0) * role_distillation_loss(role_logits, role_labels)
    if containment_pred is not None and containment_true is not None:
        total = total + w.get("contain", 0.5) * containment_distillation_loss(containment_pred, containment_true)
    if entity_anchor is not None and entity_pos is not None and entity_neg is not None:
        total = total + w.get("same_entity", 1.0) * same_entity_contrastive_loss(entity_anchor, entity_pos, entity_neg)
    if pointer_pred is not None and pointer_true is not None:
        total = total + w.get("pointer", 0.5) * pointer_observation_loss(pointer_pred, pointer_true)
    if branch_pred is not None and branch_true is not None:
        total = total + w.get("branch", 0.8) * branch_prediction_loss(branch_pred, branch_true)
    return total
