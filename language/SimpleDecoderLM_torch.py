"""SimpleDecoderLM_torch.py
PyTorch implementation of a minimal decoder-only language model, conceptually matching
`language.SimpleDecoderLM` written in NumPy. This version leverages existing PyTorch
components re-implemented earlier in the repository (token embedding with positional
encodings, multi-head self-attention, feed-forward network, etc.).

The architecture mirrors the standard Transformer decoder stack:

    Token IDs -> EmbeddingWithPosition -> N × DecoderBlock -> Linear -> Softmax

Only the forward pass is implemented given the educational focus; training helpers can
be added later.
"""

from __future__ import annotations

from typing import Optional, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from text.embedding_torch import EmbeddingWithPosition
from attention.torch_attention import MultiHeadAttention
from attention.torch_layers import LayerNormalization, FeedForward

__all__: Iterable[str]


class DecoderBlock(nn.Module):
    """Single Transformer decoder block (masked self-attention + FFN)."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = LayerNormalization(d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401, N802
        # Masked self-attention with residual & pre-norm style
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Position-wise feed-forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


class SimpleDecoderLMTorch(nn.Module):
    """Minimal decoder-only LM implemented in PyTorch."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = EmbeddingWithPosition(vocab_size, d_model, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights with token embedding for efficiency & small-scale coherence
        self.proj.weight = self.embed.token_emb.weight  # type: ignore[assignment]
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401, N802
        """Compute next-token probability distribution.

        Args:
            input_ids: ``(batch, seq_len)`` integer tensor.
            mask: optional boolean tensor ``(batch, seq_len, seq_len)``.
        Returns:
            probs: ``(batch, seq_len, vocab_size)`` probability over vocabulary.
        """
        x = self.embed(input_ids)  # (B, T, D)
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.proj(x)
        probs = self.softmax(logits)
        return probs


__all__ = (
    "DecoderBlock",
    "SimpleDecoderLMTorch",
)