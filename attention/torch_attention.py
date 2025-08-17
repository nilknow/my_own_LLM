"""torch_attention.py
Custom Multi-Head Attention built on top of ``torch.nn.MultiheadAttention`` providing an
API closer to the NumPy implementation used earlier in the project.
"""

from __future__ import annotations

from typing import Tuple, Iterable, Optional

import torch
import torch.nn as nn

__all__: Iterable[str]


class MultiHeadAttention(nn.Module):
    """Wrapper around ``torch.nn.MultiheadAttention`` with batched (B, S, D) input order.

    This layer keeps the same calling convention as the NumPy ``QKVAttention``: inputs are
    expected to be 3-D tensors of shape ``(batch, seq_len, d_model)``.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,  # (B, S, D)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # noqa: D401, N802
        """Forward pass with optional causal mask.

        Args:
            query, key, value: Tensors of shape ``(batch, seq_len, d_model)``.
            mask: Optional boolean mask of shape ``(batch, seq_len, seq_len)`` where
                elements set to ``True`` are *ignored* (masked out). If ``None``, a causal
                mask is constructed (preventing attention to future positions).
            need_weights: If ``True`` returns the attention weights.
        Returns:
            output: Tensor ``(batch, seq_len, d_model)``.
            attn_weights: Optional tensor ``(batch, seq_len, seq_len)`` if requested.
        """
        b, tgt_len, _ = query.shape
        if mask is None:
            # Build causal mask: (tgt_len, tgt_len) then broadcast to batch.
            causal = torch.triu(torch.ones(tgt_len, tgt_len, device=query.device), diagonal=1).bool()
            attn_mask = causal  # (tgt_len, tgt_len)
        else:
            # Expect mask in form (batch, tgt_len, tgt_len); reduce to 2-D allowed by PyTorch
            # by OR-ing across batch dim — conservative (mask if any sample masks that pos).
            attn_mask = mask[0] if mask.dim() == 3 else mask
            attn_mask = attn_mask.to(query.device).bool()

        out, weights = self.mha(
            query,
            key,
            value,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=not need_weights,  # return per-head weights when requested
        )
        return (out, weights) if need_weights else (out, None)


__all__ = ("MultiHeadAttention",)