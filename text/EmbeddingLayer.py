import numpy as np


class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.E = np.random.randn(vocab_size, embedding_dim) * 0.01

        self.input_ids_cache = None
        self.dE = None

    def forward(self, input_ids):
        """
        :param input_ids: shape(batch_size, sequence_length)
        :return:
        """
        self.input_ids_cache = input_ids
        return self.E[input_ids]  # Advanced Indexing

    def backward(self, dY):
        input_ids = self.input_ids_cache

        self.dE = np.zeros_like(self.E)

        for b in range(input_ids.shape[0]):
            for t in range(input_ids.shape[1]):
                np.add.at(self.dE, input_ids[b, t], dY[b, t])  # scatter add

        return None


def how_looks_like():
    E_example = np.array([
        # embedding_dim=5
        [0.005, -0.012, 0.007, 0.000, -0.004],  # ID 0 (<PAD>)
        [-0.008, 0.002, 0.015, -0.001, 0.006],  # ID 1 ('A')
        [0.001, 0.009, -0.003, 0.005, -0.000],  # ID 2 ('B')
        [0.010, -0.005, 0.002, -0.007, 0.003],  # ID 3 ('C')
        [0.004, 0.011, -0.009, 0.008, -0.006],  # ID 4 ('D')
        [0.007, -0.001, 0.004, -0.002, 0.010],  # ID 5 ('E')
        [0.003, 0.006, -0.007, 0.001, -0.005],  # ID 6
        [0.009, -0.003, 0.008, -0.004, 0.002],  # ID 7
        [-0.006, 0.000, 0.012, 0.003, -0.001],  # ID 8
        [0.002, 0.004, -0.001, 0.006, -0.008]  # ID 9
    ])
    input_ids_example = np.array([
        [2, 1, 4, 0],  # 'B', 'A', 'D', '<PAD>'
        [1, 3, 5, 0]  # 'A', 'C', 'E', '<PAD>'
    ])
    output_embeddings = np.array([
        [
            [0.001, 0.009, -0.003, 0.005, -0.000],  # ID 2 ('B')
            [-0.008, 0.002, 0.015, -0.001, 0.006],  # ID 1 ('A')
            [0.004, 0.011, -0.009, 0.008, -0.006],  # ID 4 ('D')
            [0.005, -0.012, 0.007, 0.000, -0.004]  # ID 0 ('<PAD>')
        ],
        [
            [-0.008, 0.002, 0.015, -0.001, 0.006],  # ID 1 ('A')
            [0.010, -0.005, 0.002, -0.007, 0.003],  # ID 3 ('C')
            [0.007, -0.001, 0.004, -0.002, 0.010],  # ID 5 ('E')
            [0.005, -0.012, 0.007, 0.000, -0.004]  # ID 0 ('<PAD>')
        ]
    ])


def how_to_use():
    vocab_size = 100
    embedding_dim = 64

    embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)

    input_ids_sample = np.array([
        [10, 25, 3, 80, 5],
        [2, 77, 15, 99, 10]
    ])

    embeddings = embedding_layer.forward(input_ids_sample)

    # Simulate the gradient
    dY_embedding = np.random.randn(*embeddings.shape)

    dX_embedding = embedding_layer.backward(dY_embedding)
