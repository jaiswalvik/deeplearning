import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)

# Load California housing dataset
# This dataset is used for regression tasks and contains features about housing in California.
# It includes attributes like median income, house age, average rooms, etc.
data = fetch_california_housing()
X = data.data  # Features
y = data.target  # Target
print("Number of features:", X.shape[1])

# Split the dataset into training and testing sets
# We will use 80% of the data for training and 20% for testing.
# This is a common practice to evaluate the model's performance on unseen data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check last test point
import numpy as np
print("Min and Max of last test data point:", 
      round(min(X_test[-1]), 2), round(max(X_test[-1]), 2))

# Convert data to PyTorch tensors
# PyTorch requires data to be in tensor format for training.
# We convert the features and target variables to tensors of type float32.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

print(X_train_tensor.shape, y_train_tensor.shape)

# Define a simple neural network model
# This model consists of two linear layers with ReLU activation in between.
# The input layer has 8 features (from the California housing dataset),
# and the output layer has 1 neuron for regression output.
class RegressionANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(8, 16)
        self.output = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

model = RegressionANN()

# Output layer bias
print("Initial bias of output layer:", model.output.bias.item())

# Define loss function and optimizer
# We use Mean Squared Error (MSE) as the loss function for regression tasks.
# The Adam optimizer is chosen for its efficiency in training deep learning models.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Forward pass
with torch.no_grad():
    predictions = model(X_train_tensor)
    initial_loss = criterion(predictions, y_train_tensor).item()
print("Initial loss:", initial_loss)

# Training loop
# We will train the model for 100 epochs.
# In each epoch, we perform a forward pass, compute the loss,
# perform a backward pass to compute gradients, and update the model parameters.
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Loss after 100 epochs:", loss.item())

# Define a new model with 64 neurons in the hidden layer
# This model is similar to the previous one but has a larger hidden layer.
# Increasing the number of neurons can help the model learn more complex patterns.
class RegressionANN_64(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(8, 64)
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

model_64 = RegressionANN_64()
optimizer_64 = optim.Adam(model_64.parameters(), lr=0.01)

for epoch in range(100):
    optimizer_64.zero_grad()
    predictions = model_64(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer_64.step()

print("Loss after 100 epochs with 64 neurons:", loss.item())


