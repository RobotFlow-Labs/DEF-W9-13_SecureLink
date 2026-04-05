"""Loss functions for SecureLink: Multi-Similarity Loss for metric learning.

Paper Equation (7):
    L_MS = (1/A) * sum_i [
        (1/alpha) * log(1 + sum_{j in P_i} exp(-alpha * (C_ij - mu)))
      + (1/beta)  * log(1 + sum_{r in N_i} exp( beta * (C_ir - mu)))
    ]

Where:
    - C_ij = cosine similarity between anchor i and sample j
    - P_i = positive set (same UAV), N_i = negative set (different UAV)
    - alpha=1, beta=10, mu=margin
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity Loss for deep metric learning.

    Mines positive and negative pairs from batch labels and applies
    scaled log-sum-exp objectives for both positive and negative similarities.

    Args:
        alpha: Positive pair scaling factor (default: 1.0)
        beta: Negative pair scaling factor (default: 10.0)
        margin: Similarity margin mu (default: 0.5)
        epsilon: Mining threshold for hard pair selection (default: 0.1)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 10.0,
        margin: float = 0.5,
        epsilon: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.epsilon = epsilon

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute multi-similarity loss.

        Args:
            embeddings: (batch, embedding_dim) -- L2-normalized embeddings
            labels: (batch,) -- integer UAV ID labels

        Returns:
            Scalar loss value
        """
        # Cosine similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t())  # (B, B)

        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Label masks: positive (same UAV) and negative (different UAV)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        # Exclude self-similarity on diagonal
        eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        pos_mask = labels_eq & eye_mask
        neg_mask = ~labels_eq & eye_mask

        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        num_valid_anchors = 0

        for i in range(batch_size):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            pos_sims = sim_matrix[i, pos_indices]
            neg_sims = sim_matrix[i, neg_indices]

            # Hard pair mining: keep positives harder than easiest negative - epsilon
            # and negatives harder than hardest positive + epsilon
            neg_max = neg_sims.max()
            pos_min = pos_sims.min()

            hard_pos_mask = pos_sims < (neg_max + self.epsilon)
            hard_neg_mask = neg_sims > (pos_min - self.epsilon)

            if hard_pos_mask.sum() == 0 or hard_neg_mask.sum() == 0:
                continue

            hard_pos_sims = pos_sims[hard_pos_mask]
            hard_neg_sims = neg_sims[hard_neg_mask]

            # Positive term: (1/alpha) * log(1 + sum exp(-alpha * (C_ij - margin)))
            pos_term = (1.0 / self.alpha) * torch.log(
                1.0 + torch.sum(torch.exp(-self.alpha * (hard_pos_sims - self.margin)))
            )

            # Negative term: (1/beta) * log(1 + sum exp(beta * (C_ir - margin)))
            neg_term = (1.0 / self.beta) * torch.log(
                1.0 + torch.sum(torch.exp(self.beta * (hard_neg_sims - self.margin)))
            )

            loss = loss + pos_term + neg_term
            num_valid_anchors += 1

        if num_valid_anchors > 0:
            loss = loss / num_valid_anchors

        return loss


class MultiSimilarityLossVectorized(nn.Module):
    """Vectorized multi-similarity loss for better GPU utilization.

    Faster than the loop-based version for large batches but uses more memory.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 10.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute multi-similarity loss (vectorized).

        Args:
            embeddings: (batch, embedding_dim) -- L2-normalized embeddings
            labels: (batch,) -- integer UAV ID labels

        Returns:
            Scalar loss value
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        sim_matrix = torch.mm(embeddings, embeddings.t())

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        pos_mask = labels_eq & eye_mask
        neg_mask = ~labels_eq & eye_mask

        # Large negative value for masking
        NEG_INF = -1e9

        # Positive similarities (masked)
        pos_sims = sim_matrix.clone()
        pos_sims[~pos_mask] = NEG_INF

        # Negative similarities (masked)
        neg_sims = sim_matrix.clone()
        neg_sims[~neg_mask] = NEG_INF

        # Positive term: log-sum-exp of -alpha * (pos_sim - margin)
        pos_exp = torch.exp(-self.alpha * (pos_sims - self.margin))
        pos_exp[~pos_mask] = 0.0
        pos_sum = pos_exp.sum(dim=1)
        pos_term = (1.0 / self.alpha) * torch.log(1.0 + pos_sum)

        # Negative term: log-sum-exp of beta * (neg_sim - margin)
        neg_exp = torch.exp(self.beta * (neg_sims - self.margin))
        neg_exp[~neg_mask] = 0.0
        neg_sum = neg_exp.sum(dim=1)
        neg_term = (1.0 / self.beta) * torch.log(1.0 + neg_sum)

        # Only count anchors with both positives and negatives
        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        valid = has_pos & has_neg

        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        total_loss = (pos_term[valid] + neg_term[valid]).mean()
        return total_loss
