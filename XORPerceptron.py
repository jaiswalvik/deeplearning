import numpy as np

# XOR inputs and labels
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Uncomment the next line to use XOR instead of AND or OR
y = np.array([0, 1, 1, 0])  # XOR truth table
# Uncomment the next line to use AND instead of XOR
#y = np.array([0, 0, 0, 1])  # AND truth table
# Uncomment the next line to use OR instead of XOR
#y = np.array([0, 1, 1, 1])  # OR truth table      

# Initialize weights and bias
np.random.seed(42)
weights = np.random.randn(2)
bias = np.random.randn()
lr = 0.1
epochs = 10

# Activation: Step function
def step(x):
    return 1 if x >= 0 else 0

# Training
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    errors = 0
    for i in range(len(X)):
        x_i = X[i]
        target = y[i]
        output = step(np.dot(weights, x_i) + bias)
        error = target - output
        if error != 0:
            weights += lr * error * x_i
            bias += lr * error
            errors += 1
        print(f"  Input: {x_i}, Predicted: {output}, Actual: {target}, Error: {error}")
    print(f"Total errors: {errors}\n")

# XOR Perceptron using a 2-layer network with hardcoded weights
# Step activation
def step(x):
    return 1 if x >= 0 else 0

# Perceptron unit
def perceptron(x, w, b):
    return step(np.dot(x, w) + b)

# Layer 1: two hidden units to compute intermediate logic
def hidden_layer(x):
    h1 = perceptron(x, [1, -1], -0.5)   # x1 AND NOT x2
    h2 = perceptron(x, [-1, 1], -0.5)   # NOT x1 AND x2
    return np.array([h1, h2])

# Output layer: OR gate to combine h1 and h2
def output_layer(h):
    return perceptron(h, [1, 1], -0.5)  # OR gate

# XOR model
def xor_perceptron(x):
    h = hidden_layer(x)
    y = output_layer(h)
    return y

# Test on all XOR inputs
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

print("XOR using 2-layer perceptron network (hardcoded weights):")
for x in inputs:
    print(f"Input: {x}, Output: {xor_perceptron(x)}")
