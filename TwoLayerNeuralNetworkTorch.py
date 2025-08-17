import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerNeuralNetworkTorch(nn.Module):
    """A minimal two-layer feed-forward neural network implemented with PyTorch.

    Architecture: Linear -> ReLU -> Linear (logits output).
    The Cross-Entropy loss in PyTorch (nn.CrossEntropyLoss) combines LogSoftmax + NLLLoss,
    so the forward method returns raw logits.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        x = F.relu(self.linear1(x))
        logits = self.linear2(x)
        return logits