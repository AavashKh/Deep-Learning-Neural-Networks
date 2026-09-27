import numpy as np
import pickle
from .base import Layer
from .linear_layer import LinearLayer

class Sequential(Layer):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, loss_gradient):
        gradient = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def save(self, filepath):
        model_parameters = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                model_parameters.append({'W': layer.W, 'b': layer.b})
            else:
                model_parameters.append(None)

        with open(filepath, 'wb') as f:
            pickle.dump(model_parameters, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            model_parameters = pickle.load(f)

        for layer, paramater in zip(self.layers, model_parameters):
            if parameter is not None and isinstance(layer, LinearLayer):
                layer.W = parameter['W']
                layer.b = parameter['b']
