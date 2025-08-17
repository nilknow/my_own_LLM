"""core_torch.py
PyTorch implementations of basic neural-network layers mirroring the NumPy versions found in core.py.
These lightweight wrappers mainly exist to provide a familiar API while delegating heavy lifting to
`torch.nn` primitives and automatic differentiation.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    """A thin wrapper around ``nn.Linear`` replicating the API of the NumPy version.

    Args:
        input_dim: Size of the last dimension of the input tensor.
        output_dim: Number of features produced by the layer.
        bias: If ``False``, the layer will not learn an additive bias. Default: ``True``.
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        """Compute the affine transformation *Wx + b*.

        Args:
            x: ``(..., input_dim)`` input tensor.
        Returns:
            Transformed tensor with shape ``(..., output_dim)``.
        """
        return self.linear(x)


class ReLU(nn.Module):
    """In-place or out-of-place rectified linear unit."""

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        return F.relu(x, inplace=self.inplace)


class Softmax(nn.Module):
    """Softmax along the last dimension (default behaviour mirrors NumPy implementation)."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        return F.softmax(x, dim=self.dim)


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss combining ``nn.LogSoftmax`` and ``nn.NLLLoss`` for numerical stability.

    This behaves like ``torch.nn.CrossEntropyLoss`` but is wrapped here to keep naming symmetrical
    with the NumPy version.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        """Compute the scalar cross-entropy loss.

        Args:
            logits: Raw, unnormalised model outputs of shape ``(N, C)``.
            targets: Ground-truth class indices (``LongTensor``) of shape ``(N,)`` *or* one-hot
                encodings of shape ``(N, C)``.
        Returns:
            A scalar loss value (or a vector if ``reduction='none'``).
        """
        # If targets come in one-hot format convert to indices for nn.CrossEntropyLoss.
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        return self.loss(logits, targets)

    # For symmetry with the NumPy API; autograd handles backward computation so no method needed.


__all__: Iterable[str] = (
    "LinearLayer",
    "ReLU",
    "Softmax",
    "CrossEntropyLoss",
)