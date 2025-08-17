import numpy as np

from core import Softmax


class QKVAttention:
    def __init__(self, d_model=512, d_k=64, d_v=64, n_heads=8):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.W_q = np.random.randn(d_model, d_k * n_heads) * 0.01
        self.W_k = np.random.randn(d_model, d_k * n_heads) * 0.01
        self.W_v = np.random.randn(d_model, d_v * n_heads) * 0.01
        self.W_o = np.random.randn(d_v * n_heads, d_model) * 0.01

        self.softmax = Softmax()

        self.cache = None
        self.grads = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)

        Q = Q.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.d_v).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        # Apply causal mask if none provided (prevent attending to future positions)
        if mask is None:
            seq_len = scores.shape[-1]
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)  # (seq_len, seq_len)
            scores = np.where(causal_mask, -1e9, scores)
        else:
            # User provided mask: True / 1 means masked
            scores = np.where(mask, -1e9, scores)

        attention_weights = self.softmax.forward(scores)

        attention_output = np.matmul(attention_weights, V)

        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.d_v * self.n_heads
        )

        output = np.matmul(attention_output, self.W_o)

        self.cache = {
            'query': query, 'key': key, 'value': value,
            'Q': Q, 'K': K, 'V': V,
            'attention_weights': attention_weights,
            'attention_output': attention_output,
            'mask': mask
        }

        return output, attention_weights

    def backward(self, grad_output):
        query = self.cache['query']
        key = self.cache['key']
        value = self.cache['value']
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        attention_weights = self.cache['attention_weights']
        attention_output = self.cache['attention_output']
        mask = self.cache['mask']

        batch_size = query.shape[0]

        grad_attention_output = np.matmul(grad_output, self.W_o.T)
        grad_W_o = np.matmul(attention_output.transpose(0, 2, 1), grad_output)

        grad_attention_output = grad_attention_output.reshape(
            batch_size, -1, self.n_heads, self.d_v
        ).transpose(0, 2, 1, 3)

        grad_attention_weight = np.matmul(grad_attention_output, V.transpose(0, 1, 3, 2))
        grad_V = np.matmul(attention_weights.transpose(0, 1, 3, 2), grad_attention_output)

        grad_scores = attention_weights * (
                grad_attention_weight -
                np.sum(grad_attention_weight * attention_weights, axis=-1, keepdims=True)
        )

        grad_scores = grad_scores / np.sqrt(self.d_k)

        grad_Q = np.matmul(grad_scores, K)
        grad_K = np.matmul(grad_scores.transpose(0, 1, 3, 2), Q)

        grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_k * self.n_heads)
        grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_k * self.n_heads)
        grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_v * self.n_heads)

        grad_query = np.matmul(grad_Q, self.W_q.T)
        grad_key = np.matmul(grad_K, self.W_k.T)
        grad_value = np.matmul(grad_V, self.W_v.T)

        grad_W_q = np.matmul(query.transpose(0, 2, 1), grad_Q)
        grad_W_k = np.matmul(key.transpose(0, 2, 1), grad_K)
        grad_W_v = np.matmul(value.transpose(0, 2, 1), grad_V)

        self.grads = {
            'W_q': grad_W_q,
            'W_k': grad_W_k,
            'W_v': grad_W_v,
            'W_o': grad_W_o
        }

        return grad_query, grad_key, grad_value

    def update_weights(self, learning_rate=0.001):
        self.W_q -= learning_rate * self.grads['W_q']
        self.W_k -= learning_rate * self.grads['W_k']
        self.W_v -= learning_rate * self.grads['W_v']
        self.W_o -= learning_rate * self.grads['W_o']


if __name__ == "__main__":
    # Create attention layer
    attention = QKVAttention(d_model=512, d_k=64, d_v=64, n_heads=8)

    # Sample data (batch_size=2, seq_len=10, d_model=512)
    batch_size, seq_len, d_model = 2, 10, 512
    query = np.random.randn(batch_size, seq_len, d_model)
    key = np.random.randn(batch_size, seq_len, d_model)
    value = np.random.randn(batch_size, seq_len, d_model)

    # Forward pass
    output, attn_weights = attention.forward(query, key, value)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    # Backward pass
    grad_output = np.random.randn(*output.shape)
    grad_query, grad_key, grad_value = attention.backward(grad_output)
    print(f"Gradient shapes: {grad_query.shape}, {grad_key.shape}, {grad_value.shape}")
