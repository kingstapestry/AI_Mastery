# projects/deep_learning/first_neural_net.py
"""
Lesson 36: Introduction to PyTorch & Neural Networks
Goal: Build, train, evaluate, save and load a simple neural network from scratch using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------------------------------
# 1. BASIC TENSOR OPERATIONS
# -------------------------------------------------
print("=" * 60)
print("1. BASIC TENSOR OPERATIONS")
print("=" * 60)

# Create tensors
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])          # from list
y = torch.zeros(2, 3)                                # zeros
z = torch.randn(2, 3)                                # random normal
ones = torch.ones(2, 2)

print("Tensor x:\n", x)
print("Shape of x:", x.shape)
print("Data type:", x.dtype)

# Basic operations
print("\nAddition:\n", x + x)
print("Matrix multiplication:\n", x @ x)             # or torch.matmul(x, x)
print("Element-wise multiplication:\n", x * x)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

x = x.to(device)
print("Tensor moved to device:", x.device)


# -------------------------------------------------
# 2. SIMPLE NEURAL NETWORK
# -------------------------------------------------
print("\n" + "=" * 60)
print("2. SIMPLE NEURAL NETWORK")
print("=" * 60)

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.
    Architecture: Input(2) → Hidden(4) → Output(1)
    """
    def __init__(self):
        super().__init__()                       # important!
        self.fc1 = nn.Linear(2, 4)               # input features → hidden
        self.relu = nn.ReLU()                    # activation
        self.fc2 = nn.Linear(4, 1)               # hidden → output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Create the model and move it to the correct device
model = SimpleNN().to(device)
print(model)


# -------------------------------------------------
# 3. TRAINING LOOP
# -------------------------------------------------
print("\n" + "=" * 60)
print("3. TRAINING LOOP")
print("=" * 60)

# Loss function and optimizer
criterion = nn.MSELoss()                         # Mean Squared Error (good for regression)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# -------------------------------------------------
# 4. DATA PREPARATION (XOR problem)
# -------------------------------------------------
print("\n" + "=" * 60)
print("4. DATA PREPARATION (XOR)")
print("=" * 60)

# XOR truth table
# 0 0 → 0
# 0 1 → 1
# 1 0 → 1
# 1 1 → 0

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32).to(device)

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype=torch.float32).to(device)

print("Inputs (X):\n", X)
print("Targets (y):\n", y)


# -------------------------------------------------
# Training
# -------------------------------------------------
print("\nTraining the network...")
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass + optimize
    optimizer.zero_grad()       # clear old gradients
    loss.backward()             # compute gradients
    optimizer.step()            # update weights

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")


# -------------------------------------------------
# 5. EVALUATION
# -------------------------------------------------
print("\n" + "=" * 60)
print("5. EVALUATION")
print("=" * 60)

model.eval()                    # set to evaluation mode
with torch.no_grad():           # no need to track gradients
    predictions = model(X)
    print("Predictions:\n", predictions)
    print("\nRounded predictions (0 or 1):")
    print(torch.round(predictions))


# -------------------------------------------------
# 6. SAVE & LOAD MODEL
# -------------------------------------------------
print("\n" + "=" * 60)
print("6. SAVE & LOAD MODEL")
print("=" * 60)

# Save the model
torch.save(model.state_dict(), "simple_nn.pth")
print("Model saved to 'simple_nn.pth'")

# Load the model
loaded_model = SimpleNN().to(device)
loaded_model.load_state_dict(torch.load("simple_nn.pth", map_location=device))
loaded_model.eval()

print("Model loaded successfully!")
print("Loaded model predictions:")
with torch.no_grad():
    print(loaded_model(X))


# -------------------------------------------------
# SUMMARY (Key Concepts)
# -------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY – Key Concepts")
print("=" * 60)
print("""
1. Difference between PyTorch and scikit-learn:
   - scikit-learn: High-level, fixed models (fit/predict). Great for classical ML.
   - PyTorch: Low-level, flexible, dynamic computation graphs. You define the architecture
     and the training loop yourself. Preferred for research and deep learning.

2. What nn.Module does:
   - Base class for all neural networks in PyTorch.
   - Automatically tracks parameters (weights & biases).
   - You only need to define __init__ (layers) and forward (how data flows).

3. The training loop pattern (the heart of deep learning):
   for epoch in range(epochs):
       outputs = model(inputs)          # Forward pass
       loss = criterion(outputs, targets)
       optimizer.zero_grad()            # Clear gradients
       loss.backward()                  # Backward pass (compute gradients)
       optimizer.step()                 # Update weights

This pattern is used in almost every deep learning project.
""")