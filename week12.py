import numpy as np

# ---------------------------
# Step 1: Input embeddings
# ---------------------------
X = np.array([
    [1, 0, 1],  # "The", "cat", "sat"
    [0, 1, 1]
])
print("Input Embeddings X:\n", X)

# ---------------------------
# Step 2: Transformation matrices
# ---------------------------
W_q = np.array([[1, 0],
                [0, 1]])  # Query
W_k = np.array([[0, 1],
                [1, 0]])  # Key
W_v = np.array([[1, 1],
                [0, 1]])  # Value
print("\nTransformation Matrices:")
print("W_q:\n", W_q)
print("W_k:\n", W_k)
print("W_v:\n", W_v)
# ---------------------------
# Step 3: Compute Q, K, V
# ---------------------------
Q = W_q @ X
K = W_k @ X
V = W_v @ X

print("Query matrix Q:\n", Q)
print("Key matrix K:\n", K)
print("Value matrix V:\n", V)

# ---------------------------
# Step 4: Unnormalized attention scores (Q^T K)
# ---------------------------
scores = Q.T @ K
print("\nUnnormalized Attention Scores (Q^T K):\n", scores)
print("Number of values:", scores.size)

# ---------------------------
# Step 5: Row-wise softmax to get attention weights
# ---------------------------
def softmax_rowwise(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)

A = softmax_rowwise(scores)
print("\nAttention Weights Matrix A:\n", A)

# Example: attention weight alpha_12 (first row, second column)
alpha_12 = A[0, 1]
print("\nAttention weight alpha_12:", alpha_12)

# ---------------------------
# Step 6: Compute output vectors Z
# ---------------------------
# Use Z = V @ A.T so that columns correspond to words
Z = V @ A.T
print("\nOutput vectors Z (columns = words):\n", Z)

# Sum of values in the output vector for the first word
sum_z1 = np.sum(Z[:, 0])
print("\nSum of values in the output vector for first word z1:", sum_z1)
