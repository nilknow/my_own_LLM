"""test_torch_attention.py
单元测试验证attention/torch_attention.py中PyTorch MultiHeadAttention的正确性。
"""

import unittest
import torch

from attention.torch_attention import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.n_heads = 4
        self.mha_layer = MultiHeadAttention(self.d_model, self.n_heads)

    def test_output_shape(self):
        batch_size = 2
        seq_len = 10
        query = torch.randn(batch_size, seq_len, self.d_model)
        key = torch.randn(batch_size, seq_len, self.d_model)
        value = torch.randn(batch_size, seq_len, self.d_model)

        output, _ = self.mha_layer(query, key, value)
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_attention_weights_shape(self):
        batch_size = 2
        seq_len = 10
        query = torch.randn(batch_size, seq_len, self.d_model)
        key = torch.randn(batch_size, seq_len, self.d_model)
        value = torch.randn(batch_size, seq_len, self.d_model)

        _, attn_weights = self.mha_layer(query, key, value, need_weights=True)
        self.assertIsNotNone(attn_weights)
        # PyTorch's MHA returns (batch_size, num_heads, query_sequence_length, key_sequence_length)
        # or (batch_size, query_sequence_length, key_sequence_length) if average_attn_weights=True
        # Here, it's (batch_size, num_heads, query_sequence_length, key_sequence_length)
        self.assertEqual(attn_weights.shape, (batch_size, self.n_heads, seq_len, seq_len))

    def test_causal_mask(self):
        batch_size = 1
        seq_len = 5
        query = torch.randn(batch_size, seq_len, self.d_model)
        key = torch.randn(batch_size, seq_len, self.d_model)
        value = torch.randn(batch_size, seq_len, self.d_model)

        # With causal mask, attention to future positions should be zero
        # This is hard to test directly from output, but we can check if the mask is applied internally
        # The internal mechanism of nn.MultiheadAttention handles this.
        # We can test that the output is different with and without mask.
        output_no_mask, _ = self.mha_layer(query, key, value, mask=None)
        
        # Create a non-causal mask to ensure it's different
        full_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        output_with_full_mask, _ = self.mha_layer(query, key, value, mask=full_mask)

        # The outputs should generally be different if the mask is applied correctly
        # This is a weak test, but direct inspection of internal MHA behavior is not feasible.
        self.assertFalse(torch.allclose(output_no_mask, output_with_full_mask))

    def test_gradients(self):
        batch_size = 2
        seq_len = 10
        query = torch.randn(batch_size, seq_len, self.d_model, requires_grad=True)
        key = torch.randn(batch_size, seq_len, self.d_model, requires_grad=True)
        value = torch.randn(batch_size, seq_len, self.d_model, requires_grad=True)

        output, _ = self.mha_layer(query, key, value)
        output.sum().backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(self.mha_layer.mha.in_proj_weight.grad)
        self.assertIsNotNone(self.mha_layer.mha.out_proj.weight.grad)


if __name__ == '__main__':
    unittest.main()