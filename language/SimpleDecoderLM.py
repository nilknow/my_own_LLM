import numpy as np

from text.EmbeddingLayerWithPosition import EmbeddingLayerWithPosition
from attention.TransformerBlock import TransformerBlock
from core import Softmax  # softmax exists in core.py


class SimpleDecoderLM:
    """A minimal decoder-only language model implemented in NumPy.

    Pipeline:
        Token IDs -> Token+Positional Embedding -> N x TransformerBlock -> Linear -> Softmax
    Only forward pass is provided for demonstration / testing purposes.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 2,
        max_seq_len: int = 512,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = EmbeddingLayerWithPosition(vocab_size, d_model, max_seq_len)

        self.blocks = [
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
            for _ in range(num_layers)
        ]

        # Output projection (tied weights optional)
        self.W_out = np.random.randn(d_model, vocab_size) * (d_model ** -0.5)
        self.b_out = np.zeros(vocab_size)
        self.softmax = Softmax()

    # ------------------------------------------------------------------
    # Forward (no grad)
    # ------------------------------------------------------------------
    def forward(self, input_ids: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Compute next-token distribution.

        Args:
            input_ids: (batch, seq_len) int token IDs
            mask: optional boolean mask for attention
        Returns:
            probs: (batch, seq_len, vocab_size) softmax probability over vocabulary
        """
        x = self.embed.forward(input_ids)  # (B, T, d_model)
        for blk in self.blocks:
            x = blk.forward(x, mask)

        logits = x @ self.W_out + self.b_out  # (B, T, vocab)
        probs = self.softmax.forward(logits)
        return probs


def _demo():
    batch, seq_len, vocab = 2, 10, 50
    model = SimpleDecoderLM(vocab_size=vocab, d_model=32, n_heads=4, d_ff=64, num_layers=2, max_seq_len=seq_len)
    ids = np.random.randint(0, vocab, size=(batch, seq_len))
    out = model.forward(ids)
    print("Output probs shape", out.shape, "~ should be", (batch, seq_len, vocab))
    print("Probabilities sum (first token)", out[0, 0].sum())


if __name__ == "__main__":
    _demo()