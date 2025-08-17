import numpy as np

from attention.TransformerBlock import TransformerBlock


def test_transformer_block_forward():
    """TransformerBlock forward pass should preserve (batch, seq_len, d_model) shape."""
    np.random.seed(123)

    batch, seq_len, d_model = 3, 12, 24
    x = np.random.randn(batch, seq_len, d_model)

    block = TransformerBlock(d_model=d_model, n_heads=4, d_ff=48)

    out = block.forward(x)  # causal mask applied internally

    # Shape check
    assert out.shape == (batch, seq_len, d_model)

    # Output should differ from input (very likely)
    assert not np.allclose(out, x), "Transformer block output equals input; residual + transformations seem missing"