import numpy as np

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Load parameters
files = np.load('parameters.npz')

W1 = files['W1']
b1 = files['b1'].reshape(-1, 1)
W2 = files['W2']
b2 = files['b2'].reshape(-1, 1)
W3 = files['W3']
b3 = files['b3'].reshape(-1, 1)

# Input and true label as 2D row vectors, then transpose to column vectors
x = np.array([[1, 0, 1]]).T  # shape (3, 1)
y = np.array([[0, 0, 1]]).T  # shape (3, 1)

### FORWARD PASS ###
a1 = W1 @ x + b1
print(f"a1 (preactivation layer 1):\n{a1}")
h1 = sigmoid(a1)
print(f"h1 (activation layer 1):\n{h1}")

a2 = W2 @ h1 + b2
print(f"a2 (preactivation layer 2):\n{a2}")
h2 = sigmoid(a2)
print(f"h2 (activation layer 2):\n{h2}")

a3 = W3 @ h2 + b3
print(f"a3 (preactivation output layer):\n{a3}")
h3 = softmax(a3)
print(f"h3 (output probabilities):\n{h3}")

# Cross-entropy loss
loss = -np.sum(y * np.log(h3))
print("\nLoss:", loss)

### BACKWARD PASS ###
# Output layer gradients
da3 = h3 - y  # grad wrt output pre-activation a3 (shape (3,1))
print(f"da3 (gradient wrt output pre-activation):\n{da3}")
dW3 = da3 @ h2.T
db3 = da3

# Hidden layer 2 gradients
dh2 = W3.T @ da3                  # grad wrt h2 (activation)
print(f"dh2 (gradient wrt h2 activation):\n{dh2}")
da2 = dh2 * sigmoid_derivative(h2)  # grad wrt pre-activation a2
print(f"da2 (gradient wrt pre-activation a2):\n{da2}")
dW2 = da2 @ h1.T
db2 = da2

# Hidden layer 1 gradients
dh1 = W2.T @ da2                  # grad wrt h1 (activation)
print(f"dh1 (gradient wrt h1 activation):\n{dh1}")  
da1 = dh1 * sigmoid_derivative(h1)  # grad wrt pre-activation a1
print(f"da1 (gradient wrt pre-activation a1):\n{da1}")
dW1 = da1 @ x.T
db1 = da1

# PRINT RESULTS
print("\nGradient dW3:\n", dW3)
print("\nGradient db3:\n", db3)
print("\nGradient dW2:\n", dW2)
print("\nGradient db2:\n", db2)
print("\nGradient dW1:\n", dW1)
print("\nGradient db1:\n", db1)

# Learning rate
eta = 1

# Update parameters
W1 -= eta * dW1
print("\nUpdated W1:\n", W1)
b1 -= eta * db1
print("\nUpdated b1:\n", b1)
W2 -= eta * dW2
print("\nUpdated W2:\n", W2)
b2 -= eta * db2
print("\nUpdated b2:\n", b2)
W3 -= eta * dW3
print("\nUpdated W3:\n", W3)
b3 -= eta * db3
print("\nUpdated b3:\n", b3)
# Forward pass after update
a1 = W1 @ x + b1
h1 = sigmoid(a1)

a2 = W2 @ h1 + b2
h2 = sigmoid(a2)

a3 = W3 @ h2 + b3
h3 = softmax(a3)

# New loss
new_loss = -np.sum(y * np.log(h3))
print("New loss after parameter update:", new_loss)
