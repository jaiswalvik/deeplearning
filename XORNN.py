import numpy as np

# ----- Activation functions -----
# Non-linear activation is key to the Universal Approximation Theorem (UAT)
# UAT requires activation functions like tanh/sigmoid that are non-linear and continuous
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR inputs and expected binary outputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)  # reproducibility

# ----- Network architecture -----
input_size = 2
hidden_size = 4   # UAT says a single hidden layer is sufficient; increasing size improves accuracy
output_size = 1

# Xavier initialization maintains consistent variance of activations across layers.
# This helps prevent vanishing (or exploding) gradients, especially with tanh/sigmoid activations,
# and leads to faster, more stable convergence during training.
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
b2 = np.zeros((1, output_size))

# ----- Training loop -----
epochs = 20000
learning_rate = 0.1

for epoch in range(epochs):
    # ----- Forward pass -----
    # Linear transformation followed by non-linear activation
    # This composition (Linear → Non-linear) is what enables function approximation as per UAT
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)  # Non-linear activation is essential

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # Sigmoid maps output to [0, 1] for binary classification

    # ----- Loss computation -----
    # Using MSE here, though BCE is more common for classification
    loss = np.mean((y - a2) ** 2)

    # ----- Backward pass (gradient descent) -----
    d_a2 = 2 * (a2 - y)
    d_z2 = d_a2 * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * tanh_derivative(a1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # ----- Update weights and biases -----
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# ----- Final prediction -----
print("\nFinal predictions (rounded):")
# The trained model approximates the XOR function — as guaranteed by the UAT
preds = sigmoid(np.dot(tanh(np.dot(X, W1) + b1), W2) + b2)
print(np.round(preds, 3))