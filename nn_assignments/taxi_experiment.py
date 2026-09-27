import numpy as np
import matplotlib.pyplot as plt
from neural_network import Sequential, LinearLayer, RelU, Layer

class MSELoss(Layer):
    def forward(self, y_pred, y):
        self.y_pred = y_pred
        self.y = y
        return np.mean((y_pred - y) ** 2)

    def backward(self):
        batch_size = self.y.shape[0]
        return 2 * (self.y_pred - self.y) / batch_size

print("Loading dataset... (This might take a moment)")
dataset = np.load("nyc_taxi_data.npy", allow_pickle=True).item()

X_train_df = dataset["X_train"]
X_test_df = dataset["X_test"]

X_train_df = X_train_df.select_dtypes(include=['number'])
X_test_df = X_test_df.select_dtypes(include=['number'])

X_train_full = X_train_df.to_numpy(dtype=np.float32)
y_train_full = dataset["y_train"].to_numpy(dtype=np.float32)
X_test = X_test_df.to_numpy(dtype=np.float32)
y_test = dataset["y_test"].to_numpy(dtype=np.float32)

DEV_LIMIT = 10000 
X_train_full = X_train_full[:DEV_LIMIT]
y_train_full = y_train_full[:DEV_LIMIT]

if len(y_train_full.shape) == 1:
    y_train_full = y_train_full.reshape(-1,1)
if len(y_test.shape) == 1:
    y_test = y_test.reshape(-1,1)

print("Splitting and scaling data...")
split_idx = int(0.8 * X_train_full.shape[0])
X_train, y_train = X_train_full[:split_idx], y_train_full[:split_idx]
X_val, y_val = X_train_full[split_idx:], y_train_full[split_idx:]

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_std[X_std == 0] = 1e-7 # Prevent division by 0

X_train_scaled = (X_train - X_mean) / X_std
X_val_scaled = (X_val - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std
print(f"Data ready! Training on {X_train_scaled.shape[0]} samples with {X_train_scaled.shape[1]} features.")

y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_std = y_std if y_std != 0 else 1e-7

y_train_scaled = (y_train - y_mean) / y_std
y_val_scaled = (y_val - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

def train_model(model, epochs, learning_rate):
    criteria = MSELoss()
    train_losses, val_losses = [], []

    best_val_loss = float('inf')
    steps_without_improvement = 0

    for epoch in range(epochs):
        # Forward Pass Training
        y_train_pred = model.forward(X_train_scaled)
        train_loss = criteria.forward(y_train_pred, y_train)

        # Backward Pass
        loss_gradient = criteria.backward()
        model.backward(loss_gradient)

        # Forward Pass Validation
        y_val_pred = model.forward(X_val_scaled)
        val_loss = criteria.forward(y_val_pred, y_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if steps_without_improvement >= 3:
            print(f"Stopped early at epoch: {epoch+1}")
            break

    return train_losses, val_losses

# Model Selection
input_dimension = X_train_scaled.shape[1]
configs = [
    {"name": "Config 1 : 1 Hidden Layer (8 nodes)", "layers": [8]},
    {"name": "Config 2 : 2 Hidden Layers (16, 8 nodes)", "layers": [16,8]},
    {"name": "Config 3 : 3 Hidden Layers (32, 16, 8 nodes)", "layers": [32,16,8]},
    {"name": "Config 4 : 7 Hidden Layers (512, 256, 128, 64, 32, 16, 8 nodes)", "layers": [512,256,128,64,32,16,8]}

]

best_model = None
lowest_final_val_loss = float('inf')

plt.figure(figsize=(15, 5))

for i, config in enumerate(configs):
    print(f"\nTraining {config['name']}...")
    model = Sequential()

    prev_dimension = input_dimension
    for nodes in config["layers"]:
        model.add(LinearLayer(input_dimensions=prev_dimension, output_dimensions=nodes, learning_rate=0.01))
        model.add(RelU())
        prev_dimension = nodes

    model.add(LinearLayer(input_dimensions=prev_dimension, output_dimensions=1, learning_rate=0.0001))

    train_loss, val_loss = train_model(model, epochs=1000, learning_rate=0.01)

    # Plotting
    plt.subplot(1, 4, i+1)
    plt.plot(train_loss, label="Train Loss") 
    plt.plot(val_loss, label="Val Loss")
    plt.title(config["name"])
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()

    if val_loss[-1] < lowest_final_val_loss:
        lowest_final_val_loss = val_loss[-1]
        best_model = model

plt.tight_layout()
plt.savefig("hyperparameter_tuning_plots.png")
print("\nPlots saved to 'hyperparameter_tuning_plots.png'. Displaying now...")
plt.show()

print("\n\n---Final Evaluation---\n\n")
y_test_pred = best_model.forward(X_test_scaled)
test_mse = MSELoss().forward(y_test_pred, y_test)

mean_y_test = np.mean(y_test)
ss_total = np.sum((y_test - mean_y_test) ** 2)
ss_residual = np.sum((y_test - y_test_pred) ** 2)
r2_score = 1 - (ss_residual / ss_total)

print(f"Test MSE: {test_mse:.4f}")
print(f"Test Accuracy (R-squared): {r2_score:.4f}")
