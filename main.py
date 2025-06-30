from ai.core import LinearLayer, ReLU, Softmax, CrossEntropyLoss


class TwoLayerNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, num_classes):
        # layer 1
        self.linear1 = LinearLayer(input_dim, hidden_dim)
        self.hidden_output = None
        self.relu = ReLU()
        self.relu_output = None

        # layer2
        self.linear2 = LinearLayer(hidden_dim, num_classes)
        self.logits = None
        self.softmax = Softmax()
        self.predictions = None

        self.loss_func = CrossEntropyLoss()

    def forward(self, X, y=None):
        # layer 1
        self.hidden_output = self.linear1.forward(X)
        self.relu_output = self.relu.forward(self.hidden_output)

        # layer 2
        self.logits = self.linear2.forward(self.relu_output)
        self.predictions = self.softmax.forward(self.logits)

        if y is not None:
            loss = self.loss_func.forward(self.predictions, y)
            return self.predictions, loss
        return self.predictions

    def backward(self, y):
        d_logits = self.loss_func.backward()
        d_relu_output = self.linear2.backward(d_logits)
        d_hidden_output = self.relu.backward(d_relu_output)
        return self.linear1.backward(d_hidden_output)

    def get_parameters(self):
        params = {
            'linear1_W': self.linear1.W,
            'linear1_b': self.linear1.b,
            'linear2_W': self.linear2.W,
            'linear2_b': self.linear2.b
        }
        grads = {
            'linear1_dW': self.linear1.dW,
            'linear1_db': self.linear1.db,
            'linear2_dW': self.linear2.dW,
            'linear2_db': self.linear2.db
        }
        return params, grads