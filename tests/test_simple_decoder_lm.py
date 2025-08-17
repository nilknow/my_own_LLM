import numpy as np

from language.SimpleDecoderLM import SimpleDecoderLM


def test_simple_decoder_lm_forward():
    """Sanity-check forward pass: shape, probability validity."""
    np.random.seed(42)
    batch, seq_len, vocab = 3, 5, 20

    model = SimpleDecoderLM(
        vocab_size=vocab,
        d_model=16,
        n_heads=4,
        d_ff=32,
        num_layers=2,
        max_seq_len=seq_len,
    )

    input_ids = np.random.randint(0, vocab, size=(batch, seq_len))
    out = model.forward(input_ids)  # (B, T, V)

    # 1. Shape check
    assert out.shape == (batch, seq_len, vocab), f"Output shape {out.shape} != expected {(batch, seq_len, vocab)}"

    # 2. Probabilities in [0,1]
    assert np.all(out >= 0) and np.all(out <= 1 + 1e-6), "Probabilities should lie in [0,1] range"

    # 3. Each distribution sums to 1 (within tolerance)
    sums = out.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-5), "Softmax outputs do not sum to 1 along vocab dimension"