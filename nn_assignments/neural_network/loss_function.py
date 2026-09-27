import numpy as np
from .base import Layer

class BinaryCrossEntropyLoss(Layer):
    def __init__(self):
        self.y_pred = None
        self.y = None

    def forward(self, y_pred, y):
        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # <-- To prevent log(0) errors
        self.y = y

        batch_size = y.shape[0]
        loss = -np.sum(y * np.log(self.y_pred) + (1 - y) * np.log(1 - self.y_pred)) / batch_size
        return loss

    def backward(self):
        batch_size = self.y.shape[0]

        gradient = -(np.divide(self.y, self.y_pred) - np.divide(1 - self.y, 1 - self.y_pred))
        return gradient / batch_size
