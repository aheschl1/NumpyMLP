import numpy as np


class Layer:
    def __init__(self):
        self.x = None
        self.out = None
        self.dW = None
        self.dB = None

    def __call__(self, x, *args):
        return self.forward(x, *args)

    def forward(self, x, *args):
        self.x = x

    def backward(self, grad):
        raise NotImplementedError('Do not use base class')

    def update(self, lr):
        ...


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = np.random.randn(in_features, out_features) * 0.1
        self.b = np.zeros(out_features)

    def forward(self, x, *_):
        # X is [B, in_features]
        # W is [in_features, out_features]
        # b is [out_features]
        # out is [B, out_features]
        # XW = [B, in_features] @ [in_features, out_features] = [B, out_features]
        super().forward(x)
        return (x @ self.W) + self.b

    def backward(self, grad):
        # We have Dl/Dout = grad
        # We want to compute Dl/DX, Dl/DW, Dl/Db
        # We will return Dl/DX
        # We will update W and b with Dl/DW and Dl/Db
        dW = self.x.T @ grad
        db = np.sum(grad, axis=0)
        dX = grad @ self.W.T
        self.dW = dW
        self.dB = db
        return dX

    def update(self, lr):
        if self.dW is None or self.dB is None:
            raise ValueError('No gradients to update')
        self.W -= lr * self.dW
        self.b -= lr * self.dB


class ReLU(Layer):
    def forward(self, x, *_):
        super().forward(x)
        return np.maximum(x, 0)

    def backward(self, grad):
        return grad * (self.x > 0)


class Sigmoid(Layer):
    def forward(self, x, *_):
        super().forward(x)
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)


class MSE(Layer):
    def __init__(self):
        super().__init__()
        self.y = None

    def forward(self, x, y, *_):
        super().forward(x)
        self.y = y
        return np.mean((1 / 2) * ((x - y) ** 2))

    def backward(self, grad):
        dx = grad * (self.x - self.y)
        return dx


class Tanh(Layer):
    def forward(self, x, *_):
        super().forward(x)
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)
