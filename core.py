import numpy as np


class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.random.randn(output_dim)

        self.dW = None
        self.db = None

        self.X_cache = None

    def forward(self, X):
        self.X_cache = X
        return np.dot(X, self.W) + self.b

    def backward(self, dY):
        X = self.X_cache

        self.dW = np.dot(X.T, dY)
        self.db = np.sum(dY, axis=0)

        return np.dot(dY, self.W.T)


class ReLU:
    def __init__(self):
        self.X_cache = None

    def forward(self, X):
        self.X_cache = X
        return np.maximum(0, X)

    def backward(self, dY):
        X = self.X_cache
        return dY * (X > 0)


class Softmax:
    def __init__(self):
        self.output_cache = None

    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # very interesting
        output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        self.output_cache = output  # 缓存输出
        return output

    def backward(self, dY):
        """
        This implementation won't be used in practice
        """
        softmax_output = self.output_cache
        dX = np.zeros_like(dY)

        for i in range(dY.shape[0]):
            s = softmax_output[i, :].reshape(-1, 1)
            jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
            dX[i, :] = np.dot(jacobian_matrix, dY[i, :])

        return dX


class CrossEntropyLoss:
    def __init__(self):
        self.predictions_cache = None
        self.true_labels_cache = None

    def forward(self, predictions, true_labels):
        """
        :param predictions: usually output of Softmax, shape (batch_size, num_classes)
        :param true_labels:
        :return:
        """
        num_samples = predictions.shape[0]
        self.predictions_cache = predictions

        if true_labels.ndim == 1:
            one_hot_labels = np.zeros_like(predictions)
            one_hot_labels[np.arange(num_samples), true_labels] = 1
            self.true_labels_cache = one_hot_labels
        else:
            self.true_labels_cache = true_labels

        clipped_predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)  # prevent log(0)
        return -np.sum(self.true_labels_cache * np.log(clipped_predictions)) / num_samples

    def backward(self):
        predictions = self.predictions_cache
        true_labels = self.true_labels_cache
        num_samples = predictions.shape[0]

        dX_softmax_input = (
                                       predictions - true_labels) / num_samples  # Do more research on why Softmax only works with CrossEntropyLoss
        return dX_softmax_input
