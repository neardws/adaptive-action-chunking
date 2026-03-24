"""
k-Selector: Lightweight MLP that predicts optimal action chunk size k.

Given SigLIP image features + robot state, outputs a k value from
{1, 2, 4, 8, 16} to balance latency vs. action smoothness.
"""

import torch
import torch.nn as nn
from typing import Optional

K_CANDIDATES = [1, 2, 4, 8, 16]
K_TO_IDX = {k: i for i, k in enumerate(K_CANDIDATES)}
IDX_TO_K = {i: k for i, k in enumerate(K_CANDIDATES)}


class KSelectorMLP(nn.Module):
    """
    Predicts action chunk size k from visual + state features.

    Architecture:
        [SigLIP features (1152-d)] ─┐
                                     ├─▶ MLP ─▶ softmax ─▶ k
        [robot state (32-d)]        ─┘
    """

    def __init__(
        self,
        feature_dim: int = 1152,
        state_dim: int = 32,
        hidden_dims: list[int] = [128, 64],
        n_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = feature_dim + state_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, feature_dim] — SigLIP pooled image features
            state: [B, state_dim] — robot joint states
        Returns:
            logits: [B, n_classes]
        """
        x = torch.cat([features, state], dim=-1)
        return self.mlp(x)

    def predict_k(
        self,
        features: torch.Tensor,
        state: torch.Tensor,
        temperature: float = 1.0,
    ) -> list[int]:
        """Returns a list of k values (one per batch element)."""
        logits = self.forward(features, state) / temperature
        indices = logits.argmax(dim=-1)
        return [IDX_TO_K[i.item()] for i in indices]


class KSelectorConfig:
    feature_dim: int = 1152
    state_dim: int = 32
    hidden_dims: list[int] = [128, 64]
    k_candidates: list[int] = K_CANDIDATES
    dropout: float = 0.1
