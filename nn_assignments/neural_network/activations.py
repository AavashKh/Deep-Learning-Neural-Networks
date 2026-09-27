import numpy as np
from .base import Layer

class Sigmoid(Layer):
    def __init__(self):
        self.output_cache = None

    def forward(self, X):
        self.output_cache = 1 / (1 + np.exp(-X))
        return self.output_cache

    def backward(self, upstream_gradient):
        sigmoid_gradient = self.output_cache * (1 - self.output_cache)
        return upstream_gradient * sigmoid_gradient

class RelU(Layer):
    def __init__(self):
        self.X_cache = None

    def forward(self, X):            
        self.X_cache = X
        return np.maximum(0, X)

    def backward(self, upstream_gradient):
        relu_gradient = self.X_cache > 0
        return upstream_gradient * relu_gradient

class Tanh(Layer):
    def __init__(self):
        self.output_cache = None

    def forward(self, X):
        self.output_cache = np.tanh(X)
        return self.output_cache

    def backward(self, upstream_gradient):
        tanh_gradient = 1 - (self.output_cache ** 2)
        return upstream_gradient * tanh_gradient
