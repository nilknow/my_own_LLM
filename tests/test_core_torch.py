"""test_core_torch.py
单元测试验证core_torch.py中PyTorch层的正确性。
"""

import unittest
import torch

from core_torch import LinearLayer, ReLU, Softmax, CrossEntropyLoss


class TestLinearLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.input_dim = 4
        self.output_dim = 3
        self.layer = LinearLayer(self.input_dim, self.output_dim)
        self.x = torch.randn(self.batch_size, self.input_dim)

    def test_forward_shape(self):
        output = self.layer(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_gradients_exist(self):
        output = self.layer(self.x)
        output.sum().backward()
        self.assertIsNotNone(self.layer.linear.weight.grad)
        self.assertIsNotNone(self.layer.linear.bias.grad)


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.layer = ReLU()
        self.x = torch.tensor([[-1.0, 0.0, 1.0], [0.5, -0.5, 2.0]])

    def test_forward(self):
        output = self.layer(self.x)
        expected = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.0, 2.0]])
        self.assertTrue(torch.allclose(output, expected))

    def test_gradients(self):
        x = self.x.clone().requires_grad_(True)
        output = self.layer(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.layer = Softmax()
        self.x = torch.randn(2, 3)

    def test_forward_shape(self):
        output = self.layer(self.x)
        self.assertEqual(output.shape, self.x.shape)

    def test_sum_to_one(self):
        output = self.layer(self.x)
        sums = output.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))

    def test_gradients(self):
        x = self.x.clone().requires_grad_(True)
        output = self.layer(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)


class TestCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        self.layer = CrossEntropyLoss()
        self.logits = torch.randn(2, 3)
        self.targets = torch.tensor([0, 2])

    def test_forward_scalar(self):
        loss = self.layer(self.logits, self.targets)
        self.assertEqual(loss.dim(), 0)

    def test_gradients(self):
        logits = self.logits.clone().requires_grad_(True)
        loss = self.layer(logits, self.targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()