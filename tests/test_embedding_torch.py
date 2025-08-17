"""test_embedding_torch.py
单元测试验证text/embedding_torch.py中PyTorch EmbeddingWithPosition的正确性。
"""

import unittest
import torch

from text.embedding_torch import EmbeddingWithPosition


class TestEmbeddingWithPosition(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.d_model = 64
        self.max_len = 50
        self.embedding_layer = EmbeddingWithPosition(self.vocab_size, self.d_model, self.max_len)

    def test_output_shape(self):
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        output = self.embedding_layer(input_ids)
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_positional_encoding_added(self):
        # Test that positional encoding is actually added
        batch_size = 1
        seq_len = 5
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long) # All zeros for simplicity
        
        # Get embeddings without PE (by setting PE to zeros temporarily)
        original_pe = self.embedding_layer.positional_encoding.clone()
        self.embedding_layer.positional_encoding.zero_()
        output_no_pe = self.embedding_layer(input_ids)
        
        # Restore PE and get embeddings with PE
        self.embedding_layer.positional_encoding.copy_(original_pe)
        output_with_pe = self.embedding_layer(input_ids)
        
        # The outputs should be different if PE was added
        self.assertFalse(torch.allclose(output_no_pe, output_with_pe))

    def test_gradients(self):
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        
        # Embedding layer's weights should have gradients
        output = self.embedding_layer(input_ids)
        output.sum().backward()
        self.assertIsNotNone(self.embedding_layer.token_emb.weight.grad)


if __name__ == '__main__':
    unittest.main()