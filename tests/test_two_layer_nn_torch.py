import torch
from TwoLayerNeuralNetworkTorch import TwoLayerNeuralNetworkTorch


def test_two_layer_nn_torch_forward_backward():
    """Ensure the PyTorch 2-layer NN produces logits of correct shape and gradients."""
    torch.manual_seed(42)

    batch_size, input_dim, hidden_dim, num_classes = 8, 20, 32, 10
    model = TwoLayerNeuralNetworkTorch(input_dim, hidden_dim, num_classes)

    x = torch.randn(batch_size, input_dim, requires_grad=True)
    y = torch.randint(0, num_classes, (batch_size,))

    criterion = torch.nn.CrossEntropyLoss()

    logits = model(x)
    assert logits.shape == (batch_size, num_classes)

    loss = criterion(logits, y)
    loss.backward()

    # Check gradients exist
    assert model.linear1.weight.grad is not None
    assert model.linear2.weight.grad is not None

    # Simple optimizer step to ensure parameters update without errors
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = model.linear1.weight.clone().detach()
    optimizer.step()
    after = model.linear1.weight.clone().detach()
    assert not torch.allclose(before, after)