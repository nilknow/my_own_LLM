import numpy as np

from core import LinearLayer, ReLU


class FeedForward:
    def __init__(self, d_model: int, d_ff: int = 2048):
        """Constructor

        Args:
            d_model: Input and output dimensionality (model dimension)
            d_ff: Hidden layer dimensionality (usually 2-4 * d_model)
        """
        self.linear1 = LinearLayer(d_model, d_ff)
        self.relu = ReLU()
        self.linear2 = LinearLayer(d_ff, d_model)

        # cache for backward
        self._cache_input = None

    # ---------------------------------------------------------------------
    # Forward / Backward
    # ---------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        # Save input for backward (residual path might need it)
        self._cache_input = x

        # Merge batch & sequence dims for matrix ops, then reshape back
        b, s, d = x.shape
        x2d = x.reshape(-1, d)  # (b*s, d_model)

        out1 = self.linear1.forward(x2d)
        out2 = self.relu.forward(out1)
        out3 = self.linear2.forward(out2)

        return out3.reshape(b, s, d)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass.
        Args:
            grad_output: Gradient w.r.t. output of shape (batch, seq_len, d_model)
        Returns:
            Gradient w.r.t. input of same shape
        """
        b, s, d = grad_output.shape
        grad2d = grad_output.reshape(-1, d)

        grad_l2 = self.linear2.backward(grad2d)
        grad_relu = self.relu.backward(grad_l2)
        grad_l1 = self.linear1.backward(grad_relu)

        return grad_l1.reshape(b, s, d)

    # ------------------------------------------------------------------
    # Optimisation helpers
    # ------------------------------------------------------------------
    def update_weights(self, lr: float = 1e-3):
        """Simple SGD step for all sub-layers"""
        self.linear1.W -= lr * self.linear1.dW
        self.linear1.b -= lr * self.linear1.db
        self.linear2.W -= lr * self.linear2.dW
        self.linear2.b -= lr * self.linear2.db