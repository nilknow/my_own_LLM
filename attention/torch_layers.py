"""torch_layers.py
Collection of PyTorch re-implementations for components previously written in NumPy.
These modules closely mirror their NumPy counterparts while embracing ``torch.nn``
primitives and autograd.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__: Iterable[str]

# -----------------------------------------------------------------------------
# Positional Encoding (sinusoidal, no learnable params)
# -----------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding identical to the formulation in "Attention Is All You Need".

    Adds a fixed position-dependent embedding to the input tensor. The implementation is
    vectorised and pre-computes the encoding up to ``max_len``; the matrix is stored as a
    non-trainable buffer so it moves automatically with ``.to(device)``.
    """

    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        """Add positional encoding.

        Args:
            x: Tensor of shape ``(batch, seq_len, d_model)``.
        Returns:
            Same shape tensor with positional information added.
        """
        seq_len = x.size(1)
        # pe: (seq_len, d_model) -> (1, seq_len, d_model) for broadcast
        return x + self.pe[:seq_len].unsqueeze(0)


# -----------------------------------------------------------------------------
# Layer Normalisation (wrapper for nn.LayerNorm to keep naming symmetrical)
# -----------------------------------------------------------------------------

class LayerNormalization(nn.Module):
    """Thin wrapper around ``torch.nn.LayerNorm`` using last dimension normalisation."""

    def __init__(self, d_model: int, epsilon: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        return self.norm(x)


# -----------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# -----------------------------------------------------------------------------

class FeedForward(nn.Module):
    """2-layer MLP with ReLU activation, applied independently to each sequence position."""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


__all__ = ("PositionalEncoding", "LayerNormalization", "FeedForward")