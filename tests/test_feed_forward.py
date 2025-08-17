import numpy as np

from attention.FeedForward import FeedForward


def test_feed_forward_forward_backward():
    """FeedForward layer should produce correct output shape and backward gradients."""
    np.random.seed(42)

    batch, seq_len, d_model, d_ff = 4, 10, 16, 32
    x = np.random.randn(batch, seq_len, d_model)

    ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    # Forward pass
    out = ffn.forward(x)
    assert out.shape == (batch, seq_len, d_model)

    # Backward pass with random gradient
    grad_out = np.random.randn(*out.shape)
    grad_in = ffn.backward(grad_out)
    assert grad_in.shape == x.shape

    # Ensure gradients are computed for parameters
    assert ffn.linear1.dW is not None and ffn.linear1.db is not None
    assert ffn.linear2.dW is not None and ffn.linear2.db is not None

    # Parameter update should change weights
    W1_before = ffn.linear1.W.copy()
    ffn.update_weights(lr=1e-3)
    assert not np.array_equal(W1_before, ffn.linear1.W)