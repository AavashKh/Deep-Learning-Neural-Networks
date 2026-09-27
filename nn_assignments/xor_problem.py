'''
Training with Tanh was easier and faster for hidden layers.
'''

import numpy as np
import matplotlib.pyplot as plt
from neural_network import Sequential, LinearLayer, Sigmoid, Tanh, BinaryCrossEntropyLoss

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

def train_xor(hidden_activation, epochs=10000, learning_rate=0.5):
    model = Sequential()
    
    model.add(LinearLayer(input_dimensions=2, output_dimensions=2, learning_rate=learning_rate))
    model.add(hidden_activation())
    
    model.add(LinearLayer(input_dimensions=2, output_dimensions=1, learning_rate=learning_rate))
    model.add(Sigmoid())
    
    criterion = BinaryCrossEntropyLoss()
    loss_history = []
    
    for epoch in range(epochs):
        # Forward Pass
        y_pred = model.forward(X)
        loss = criterion.forward(y_pred, y)
        loss_history.append(loss)
        
        # Backward Pass
        loss_grad = criterion.backward()
        model.backward(loss_grad)

        if loss < 0.01:
            print(f"Converged at epoch {epoch} with loss {loss:.4f}")
            break
            
    return model, loss_history

print("Training with Sigmoid Activation")

np.random.seed(42) 
sigmoid_model, sigmoid_losses = train_xor(Sigmoid, epochs=10000, learning_rate=1.0)
print(f"Final predictions (Sigmoid) : \n{sigmoid_model.forward(X)}\n")

print("Training with Tanh Activation")
np.random.seed(42) 
tanh_model, tanh_losses = train_xor(Tanh, epochs=10000, learning_rate=1.0)
print(f"Final predictions (Tanh) : \n{tanh_model.forward(X)}\n")


save_filepath = "XOR_solved.w"
tanh_model.save(save_filepath)
print(f"Weights successfully saved to {save_filepath}")
