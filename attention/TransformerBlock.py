import numpy as np

from attention.QKVAttention import QKVAttention
from attention.FeedForward import FeedForward
from attention.LayerNormalization import layer_norm


class TransformerBlock:
    """Minimal Transformer decoder block (no cross-attention).

    Structure:  
        x -> Masked Multi-Head Self-Attention -> Add & Norm -> FeedForward -> Add & Norm

    Only forward pass is implemented for study purposes. Back-prop can be added similarly by
    delegating to sub-modules.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048,
                 d_k: int | None = None, d_v: int | None = None):
        d_k = d_k or d_model // n_heads
        d_v = d_v or d_model // n_heads

        self.self_attn = QKVAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

        # caches for backward if required later
        self._cache_attn_out = None
        self._cache_ffn_out = None

    # ---------------------------------------------------------------------
    # Forward (inference / demo)
    # ---------------------------------------------------------------------
    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            mask: optional boolean mask for attention (True = masked)
        Returns:
            Tensor same shape as x
        """
        # Self-attention + residual + norm
        attn_out, _ = self.self_attn.forward(x, x, x, mask)
        res1 = x + attn_out
        norm1 = layer_norm(res1)
        self._cache_attn_out = norm1

        # Position-wise FFN + residual + norm
        ffn_out = self.ffn.forward(norm1)
        res2 = norm1 + ffn_out
        norm2 = layer_norm(res2)
        self._cache_ffn_out = norm2

        return norm2

    # ------------------------------------------------------------------
    # (Optional) Backward & update helpers could be added similarly
    # ------------------------------------------------------------------

    def parameters(self):
        """Return list of parameter arrays for simple introspection/train loops."""
        return [
            self.self_attn.W_q, self.self_attn.W_k, self.self_attn.W_v, self.self_attn.W_o,
            self.ffn.linear1.W, self.ffn.linear1.b,
            self.ffn.linear2.W, self.ffn.linear2.b,
        ]