import numpy as np

# Load parameters
parameters = np.load("parameters1.npz")   # adjust path if needed
U = parameters.get("U")
W = parameters.get("W")
V = parameters.get("V")

# One-hot encodings
chars = ['l','e','a','r','n']
one_hot = {
    'l': np.array([[1],[0],[0],[0],[0]]),
    'e': np.array([[0],[1],[0],[0],[0]]),
    'a': np.array([[0],[0],[1],[0],[0]]),
    'r': np.array([[0],[0],[0],[1],[0]]),
    'n': np.array([[0],[0],[0],[0],[1]])
}

# Training sequence ("learn")
inputs = [one_hot[c] for c in "learn"]
targets = [one_hot[c] for c in "earn"] + [np.zeros((5,1))]   # last target = zero vector

# Initial state
s0 = np.zeros((U.shape[0],1))

# Activation functions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Forward pass
s = {0: s0}
y_hats, loss = {}, 0
for t in range(len(inputs)):
    s[t+1] = np.tanh(U @ inputs[t] + W @ s[t])   # no biases
    z = V @ s[t+1]
    y_hats[t] = softmax(z)
    loss += -float(targets[t].T @ np.log(y_hats[t] + 1e-12))

print("Loss at t=1 (L1):", -float(targets[0].T @ np.log(y_hats[0] + 1e-12)))
print("Total Loss L(θ):", loss)

# Backpropagation Through Time (BPTT)
dU = np.zeros_like(U)
dW = np.zeros_like(W)
dV = np.zeros_like(V)
ds_next = np.zeros_like(s[0])

for t in reversed(range(len(inputs))):
    dy = y_hats[t] - targets[t]     # cross-entropy derivative
    dV += dy @ s[t+1].T
    ds = V.T @ dy + ds_next
    ds_raw = (1 - s[t+1]**2) * ds
    dU += ds_raw @ inputs[t].T
    dW += ds_raw @ s[t].T
    ds_next = W.T @ ds_raw

print("Sum of elements in ∂L/∂W:", dW.sum())
max_val = dW.max()
max_index = tuple(int(i) for i in np.unravel_index(np.argmax(dW), dW.shape))
print("Max value of ∂L/∂W:", max_val, "at index", max_index)

lr = 0.1

U -= lr * dU
W -= lr * dW
V -= lr * dV

x = one_hot['l']
s_t = np.zeros((U.shape[0], 1))
generated_sequence = []

for _ in range(5):  # next 5 characters
    s_t = np.tanh(U @ x + W @ s_t)
    y_hat = softmax(V @ s_t)
    idx = np.argmax(y_hat)
    next_char = ['l','e','a','r','n'][idx]
    generated_sequence.append(next_char)
    x = one_hot[next_char]

print("Generated sequence:", ''.join(generated_sequence))  # enlrn
