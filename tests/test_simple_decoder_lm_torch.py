import torch

from language.SimpleDecoderLM_torch import SimpleDecoderLMTorch


def test_simple_decoder_lm_torch_forward():
    """Sanity-check forward pass: shape and probability validity."""
    torch.manual_seed(42)
    batch, seq_len, vocab = 3, 5, 20

    model = SimpleDecoderLMTorch(
        vocab_size=vocab,
        d_model=16,
        n_heads=4,
        d_ff=32,
        num_layers=2,
        max_seq_len=seq_len,
    )

    input_ids = torch.randint(0, vocab, (batch, seq_len))
    out = model(input_ids)  # (B, T, V)

    # 1. Shape check
    assert out.shape == (batch, seq_len, vocab)

    # 2. Probabilities in [0,1]
    assert torch.all(out >= 0) and torch.all(out <= 1 + 1e-6)

    # 3. Each distribution sums to 1 (within tolerance)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_simple_decoder_lm_torch_gradients():
    """Ensure gradients flow through the model."""
    torch.manual_seed(0)
    batch, seq_len, vocab = 2, 4, 10
    model = SimpleDecoderLMTorch(
        vocab_size=vocab,
        d_model=8,
        n_heads=2,
        d_ff=16,
        num_layers=1,
        max_seq_len=seq_len,
    )

    input_ids = torch.randint(0, vocab, (batch, seq_len))
    out = model(input_ids)
    loss = out.mean()
    loss.backward()

    # Check gradients exist on embedding table and output projection (tied weights)
    assert model.embed.token_emb.weight.grad is not None
    assert model.proj.weight.grad is not None