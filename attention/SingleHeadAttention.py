import numpy as np

from core import Softmax


class SingleHeadAttention:
    def __init__(self, d_model=512):
        self.d_model = d_model

        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

        self.softmax = Softmax()
        self.cache = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = np.matmul(query, self.W_q)  # (batch, seq_len, d_model)
        K = np.matmul(key, self.W_k)  # (batch, seq_len, d_model)
        V = np.matmul(value, self.W_v)  # (batch, seq_len, d_model)

        # Q: (batch, seq_len, d_model)
        # K^T: (batch, d_model, seq_len)
        # scores: (batch, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_model)

        # Apply causal mask (prevent attending to future positions)
        if mask is None:
            seq_len = scores.shape[-1]
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores = np.where(causal_mask, -1e9, scores)
        else:
            # User-provided mask (1 = keep, 0 = mask)
            scores = np.where(mask == 0, -1e9, scores)

        attention_weights = self.softmax.forward(scores)

        # attention_weights: (batch, seq_len, seq_len)
        # V: (batch, seq_len, d_model)
        # output: (batch, seq_len, d_model)
        attention_output = np.matmul(attention_weights, V)

        output = np.matmul(attention_output, self.W_o)

        # 缓存前向传播结果用于反向传播
        self.cache = {
            'query': query,
            'key': key,
            'value': value,
            'Q': Q,
            'K': K,
            'V': V,
            'scores': scores,
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
        scores = self.cache['scores']

        batch_size = query.shape[0]

        # grad_output: (batch, seq_len, d_model)
        # attention_output: (batch, seq_len, d_model)
        grad_W_o = np.matmul(attention_output.transpose(0, 2, 1), grad_output)
        grad_W_o = np.sum(grad_W_o, axis=0)

        # grad_output: (batch, seq_len, d_model)
        # W_o: (d_model, d_model)
        grad_attention_output = np.matmul(grad_output, self.W_o.T)

        # grad_attention_output: (batch, seq_len, d_model)
        # V: (batch, seq_len, d_model)
        grad_attention_weights = np.matmul(grad_attention_output, V.transpose(0, 2, 1))

        # attention_weights: (batch, seq_len, seq_len)
        # grad_attention_output: (batch, seq_len, d_model)
        grad_V = np.matmul(attention_weights.transpose(0, 2, 1), grad_attention_output)

        grad_scores = attention_weights * (
                grad_attention_weights -
                np.sum(grad_attention_weights * attention_weights, axis=-1, keepdims=True)
        )

        grad_scores = grad_scores / np.sqrt(self.d_model)

        # grad_scores: (batch, seq_len, seq_len)
        # K: (batch, seq_len, d_model)
        grad_Q = np.matmul(grad_scores, K)

        # grad_scores^T: (batch, seq_len, seq_len)
        # Q: (batch, seq_len, d_model)
        grad_K = np.matmul(grad_scores.transpose(0, 2, 1), Q)

        # Q = query @ W_q -> grad_W_q = query^T @ grad_Q
        grad_W_q = np.matmul(query.transpose(0, 2, 1), grad_Q)
        grad_W_q = np.sum(grad_W_q, axis=0)  # 聚合批次维度

        # K = key @ W_k -> grad_W_k = key^T @ grad_K
        grad_W_k = np.matmul(key.transpose(0, 2, 1), grad_K)
        grad_W_k = np.sum(grad_W_k, axis=0)  # 聚合批次维度

        # V = value @ W_v -> grad_W_v = value^T @ grad_V
        grad_W_v = np.matmul(value.transpose(0, 2, 1), grad_V)
        grad_W_v = np.sum(grad_W_v, axis=0)  # 聚合批次维度

        # grad_Q: (batch, seq_len, d_model)
        # W_q: (d_model, d_model)
        grad_query = np.matmul(grad_Q, self.W_q.T)

        # grad_K: (batch, seq_len, d_model)
        # W_k: (d_model, d_model)
        grad_key = np.matmul(grad_K, self.W_k.T)

        # grad_V: (batch, seq_len, d_model)
        # W_v: (d_model, d_model)
        grad_value = np.matmul(grad_V, self.W_v.T)

        # 存储梯度供参数更新使用
        self.grads = {
            'W_q': grad_W_q,
            'W_k': grad_W_k,
            'W_v': grad_W_v,
            'W_o': grad_W_o
        }

        return grad_query, grad_key, grad_value

    def update_weights(self, learning_rate=0.001):
        if hasattr(self, 'grads'):
            self.W_q -= learning_rate * self.grads['W_q']
            self.W_k -= learning_rate * self.grads['W_k']
            self.W_v -= learning_rate * self.grads['W_v']
            self.W_o -= learning_rate * self.grads['W_o']
