import os
import numpy as np
from neural_network import (
    Sequential, 
    LinearLayer, 
    RelU, 
    Sigmoid, 
    BinaryCrossEntropyLoss
)

# Set a random seed so the results are exactly the same every time you test
np.random.seed(42)

# ==========================================
# PART 1: FORWARD, LOSS, AND BACKWARD PASS
# ==========================================
print("--- PART 1: TRAINING PIPELINE TEST ---")

# Dummy data: 4 samples, 3 input features
X = np.array([
    [0.1, 0.2, 0.7],
    [0.8, 0.1, 0.3],
    [0.4, 0.9, 0.2],
    [0.9, 0.8, 0.1]
])

# Dummy targets: 4 binary outputs (shape: 4x1)
y = np.array([
    [1],
    [0],
    [1],
    [0]
])

# Build the architecture: 3 -> 4 (ReLU) -> 1 (Sigmoid)
model = Sequential()
model.add(LinearLayer(input_dimensions=3, output_dimensions=4, learning_rate=0.1))
model.add(RelU())
model.add(LinearLayer(input_dimensions=4, output_dimensions=1, learning_rate=0.1))
model.add(Sigmoid())

criterion = BinaryCrossEntropyLoss()

# Capture the starting weights of the first layer BEFORE the forward pass
linear_layer_1 = model.layers[0]
initial_weights = linear_layer_1.W.copy()

# 1. Forward pass
y_pred = model.forward(X)
print("Forward pass successful. Output shape:", y_pred.shape)

# 2. Compute Loss
loss = criterion.forward(y_pred, y)
print(f"BCE Loss: {loss:.5f}")

# 3. Backward pass
loss_grad = criterion.backward()
model.backward(loss_grad)

# 4. Verify weight updates
updated_weights = linear_layer_1.W

if not np.array_equal(initial_weights, updated_weights):
    print("SUCCESS: Weights were updated! The Sequential backward pass is working.\n")
else:
    print("FAILURE: Weights did not change. Check your backward methods.\n")


# ==========================================
# PART 2: SAVING AND LOADING
# ==========================================
print("--- PART 2: SAVING AND LOADING TEST ---")

save_path = "test_weights.w"

# 1. Get a baseline prediction from the trained model to compare later
original_pred = model.forward(X)

# 2. Save the trained model
model.save(save_path)
print(f"Model saved to {save_path}")

# 3. Create a brand new model with the exact same architecture
# (Its weights will be completely randomized upon initialization)
loaded_model = Sequential()
loaded_model.add(LinearLayer(input_dimensions=3, output_dimensions=4))
loaded_model.add(RelU())
loaded_model.add(LinearLayer(input_dimensions=4, output_dimensions=1))
loaded_model.add(Sigmoid())

# 4. Load the weights into the new model
loaded_model.load(save_path)
print("Model loaded successfully.")

# 5. Verify the weights and biases match exactly
layer1_W_match = np.allclose(model.layers[0].W, loaded_model.layers[0].W)
layer1_b_match = np.allclose(model.layers[0].b, loaded_model.layers[0].b)

print(f"Layer 1 Weights Match: {layer1_W_match}")
print(f"Layer 1 Biases Match:  {layer1_b_match}")

# 6. Verify the outputs match exactly
loaded_pred = loaded_model.forward(X)
output_match = np.allclose(original_pred, loaded_pred)

print(f"Predictions Match:     {output_match}")

if layer1_W_match and layer1_b_match and output_match:
    print("SUCCESS: Save and load functionality is working perfectly.\n")
else:
    print("FAILURE: Loaded weights or outputs do not match the original model.\n")

