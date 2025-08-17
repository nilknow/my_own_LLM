"""embedding_torch.py
PyTorch implementation of token embedding **plus** sinusoidal positional encoding,
functionally equivalent to `EmbeddingLayerWithPosition` defined in NumPy.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

__all__: Iterable[str]


class EmbeddingWithPosition(nn.Module):
    """Token embedding table combined with fixed sinusoidal positional encoding.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Dimensionality of embeddings.
        max_len: Maximum sequence length supported.
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("positional_encoding", self._build_pe(max_len, d_model))

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        """Lookup token embeddings and add positional encodings.

        Args:
            input_ids: ``(batch, seq_len)`` tensor of token indices.
        Returns:
            Embedded representation of shape ``(batch, seq_len, d_model)``.
        """
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_emb(input_ids)
        pos_emb = self.positional_encoding[:seq_len].unsqueeze(0)
        return token_emb + pos_emb


__all__ = ("EmbeddingWithPosition",)