import numpy as np
from .base import Layer

class LinearLayer(Layer):
    def __init__(self, input_dimensions, output_dimensions, learning_rate=0.01):
        self.W = np.random.randn(input_dimensions, output_dimensions) * 0.01 # to set the value close to 0
        self.b = np.zeros((1, output_dimensions))
        self.alpha = learning_rate
        self.X_cache = None

    def forward(self, X):
        self.X_cache = X

        return np.dot(X, self.W) + self.b

    def backward(self, upstream_gradient):
        dW = np.dot(self.X_cache.T, upstream_gradient)
        db = np.sum(upstream_gradient, axis=0, keepdims=True)
        dX = np.dot(upstream_gradient, self.W.T)

        self.W -= self.alpha * dW
        self.b -= self.alpha * db

        return dX
