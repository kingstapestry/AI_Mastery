import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


"""
LESSON 36: Introduction to PyTorch & Neural Networks (Phase 4 Start)

Goal:
    Learn the basics of PyTorch - the main deep learning framework used by many frontier AI labs.
"""

# ==================== 1. BASIC TENSOR OPERATIONS ====================
print("=== 1. Tensors in PyTorch ===")

# Tensors are like NumPy arrays but can run on GPU and support automatic differentiation
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"Tensor x: {x}")
print(f"Tensor y: {y}")
print(f"Addition: {x + y}")
print(f"Multiplication: {x * y}")

# Moving to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

x = x.to(device)
print(f"Tensor moved to {device}")


# ==================== 2. SIMPLE NEURAL NETWORK ====================
print("\n=== 2. Creating a Neural Network ===")

class SimpleNet(nn.Module):
    """
    A simple neural network.
    nn.Module is the base class for all neural networks in PyTorch.
    """
    def __init__(self):
        super().__init__()                          # Required for nn.Module
        self.layer1 = nn.Linear(2, 10)              # Input: 2 features → Hidden: 10 neurons
        self.layer2 = nn.Linear(10, 1)              # Hidden: 10 → Output: 1 value
        self.relu = nn.ReLU()                       # Activation function
    
    def forward(self, x):
        """
        Defines how data flows through the network.
        This is where the magic happens.
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# Create the model
model = SimpleNet()
print("Model created successfully!")


# ==================== 3. TRAINING LOOP ====================
print("\n=== 3. Training the Network ===")

# Synthetic data (XOR-like problem)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Loss function and optimizer
criterion = nn.MSELoss()                        # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adam optimizer

# Training loop
for epoch in range(1000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass (backpropagation)
    optimizer.zero_grad()                       # Clear previous gradients
    loss.backward()                             # Compute gradients
    optimizer.step()                            # Update weights
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# ==================== 4. EVALUATION ====================
print("\n=== 4. Final Evaluation ===")

with torch.no_grad():                           # Disable gradient calculation for inference
    predictions = model(X)
    print("Predictions:")
    for i, pred in enumerate(predictions):
        print(f"Input: {X[i].tolist()} → Predicted: {pred.item():.4f} → Target: {y[i].item()}")


# ==================== 5. SAVE & LOAD MODEL ====================
print("\n=== 5. Saving & Loading Model ===")

# Save model
torch.save(model.state_dict(), "projects\deep_learning\simple_net.pth")
print("Model saved as 'projects\deep_learning\simple_net.pth'")

# Load model
loaded_model = SimpleNet()
loaded_model.load_state_dict(torch.load("projects\deep_learning\simple_net.pth"))
print("Model loaded successfully!")


# ==================== SUMMARY ====================
"""
Key Takeaways from This Lesson:

- PyTorch is dynamic and Pythonic (unlike TensorFlow's static graphs)
- nn.Module is the base class for all neural networks
- forward() defines how data flows through the network
- Training loop: forward → loss → backward → optimize
- Tensors can run on CPU or GPU
- Saving/loading models is simple with state_dict()

This is the foundation for all deep learning in PyTorch.
"""