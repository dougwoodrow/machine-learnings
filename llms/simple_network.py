import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ====================== PART 1: NumPy MLP from Scratch ======================

def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


class NumpyMLP:
    def __init__(self, input_size=64, hidden_size=32, output_size=10):
        # He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)  # <-- Activation 1
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)  # <-- Activation 2 (output)
        return self.a2

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def train(self, X, y, epochs=100, lr=0.01):
        for epoch in range(epochs):
            # Forward
            output = self.forward(X)
            loss = -np.mean(y * np.log(output + 1e-8))

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

                # Track activations
                print(f"Hidden activation sparsity: {(self.a1 == 0).mean():.1%} zeros")
        return self


# ====================== PART 2: Load Data ======================
digits = load_digits()
X = digits.data / 16.0  # normalize
y = digits.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# ====================== PART 3: Train NumPy Model & Visualize ======================
model_np = NumpyMLP()
model_np.train(X_train, y_train, epochs=200, lr=0.01)

# Visualize activations from last forward pass
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(model_np.a1.flatten(), bins=50, kde=True)
plt.title("Hidden Layer Activations (ReLU)")

plt.subplot(1, 2, 2)
sns.histplot(model_np.a2.flatten(), bins=50, kde=True)
plt.title("Output Layer Activations (Softmax)")
plt.show()