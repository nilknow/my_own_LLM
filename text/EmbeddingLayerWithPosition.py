import numpy as np

from attention.PositionEncoding import get_sinusoidal_position_encoding


class EmbeddingLayerWithPosition:
    def __init__(self, vocab_size, embedding_dim, max_seq_len=512):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.E = np.random.randn(vocab_size, embedding_dim) * 0.01

        self.pe = get_sinusoidal_position_encoding(max_seq_len, embedding_dim)

        # Cache for backpropagation
        self.input_ids_cache = None
        self.dE = None
        self.seq_len_cache = None

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        self.input_ids_cache = input_ids
        self.seq_len_cache = seq_len

        token_emb = self.E[input_ids]

        pos_emb = self.pe[np.newaxis, :seq_len, :]

        return token_emb + pos_emb

    def backward(self, dY):
        input_ids = self.input_ids_cache
        seq_len = self.seq_len_cache

        self.dE = np.zeros_like(self.E)

        # Flatten input_ids and gradients for efficient processing
        flat_input_ids = input_ids.flatten()
        flat_dY = dY.reshape(-1, self.embedding_dim)

        # Accumulate gradients for each token embedding using scatter-add operation
        np.add.at(self.dE, flat_input_ids, flat_dY)

        # No gradient is passed to input_ids as they are discrete indices
        return None


def example_usage():
    # Parameters
    vocab_size = 10000
    embedding_dim = 256
    max_seq_len = 128
    batch_size = 2
    seq_len = 16
    
    # Create random input IDs (token indices)
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    
    # Initialize embedding layer with positional encoding
    embedding_layer = EmbeddingLayerWithPosition(vocab_size, embedding_dim, max_seq_len)
    
    # Forward pass
    embeddings = embedding_layer.forward(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    
    # Simulate gradient from next layer
    dY = np.random.randn(*embeddings.shape)
    
    # Backward pass
    embedding_layer.backward(dY)
    print(f"Embedding gradients shape: {embedding_layer.dE.shape}")
    
    return embeddings


if __name__ == '__main__':
    example_usage()
