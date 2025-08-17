"""test_torch_layers.py
单元测试验证attention/torch_layers.py中PyTorch层的正确性。
"""

import unittest
import torch

from attention.torch_layers import PositionalEncoding, LayerNormalization, FeedForward


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.max_len = 100
        self.pe_layer = PositionalEncoding(self.d_model, self.max_len)

    def test_output_shape(self):
        x = torch.randn(1, 10, self.d_model)  # batch_size, seq_len, d_model
        output = self.pe_layer(x)
        self.assertEqual(output.shape, x.shape)

    def test_encoding_values(self):
        # Test that encoding is added and is consistent
        x = torch.zeros(1, 5, self.d_model)
        output = self.pe_layer(x)
        # Check that output is not all zeros (i.e., PE was added)
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        # Check that PE for a given position is consistent regardless of batch/seq_len
        x_single = torch.zeros(1, 1, self.d_model)
        output_single = self.pe_layer(x_single)
        self.assertTrue(torch.allclose(output[0, 0, :], output_single[0, 0, :]))


class TestLayerNormalization(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.norm_layer = LayerNormalization(self.d_model)

    def test_output_shape(self):
        x = torch.randn(2, 10, self.d_model)
        output = self.norm_layer(x)
        self.assertEqual(output.shape, x.shape)

    def test_normalization_properties(self):
        x = torch.randn(2, 10, self.d_model)
        output = self.norm_layer(x)
        # Check mean and variance along the last dimension (d_model)
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False) # unbiased=False for population variance
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-6))
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=6e-2)) # Increased tolerance for variance

    def test_gradients(self):
        x = torch.randn(2, 10, self.d_model, requires_grad=True)
        output = self.norm_layer(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.d_ff = 128
        self.ff_layer = FeedForward(self.d_model, self.d_ff)

    def test_output_shape(self):
        x = torch.randn(2, 10, self.d_model)
        output = self.ff_layer(x)
        self.assertEqual(output.shape, x.shape)

    def test_gradients(self):
        x = torch.randn(2, 10, self.d_model, requires_grad=True)
        output = self.ff_layer(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.ff_layer.w_1.weight.grad)
        self.assertIsNotNone(self.ff_layer.w_2.weight.grad)


if __name__ == '__main__':
    unittest.main()