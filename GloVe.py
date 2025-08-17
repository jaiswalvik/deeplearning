import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------
# 1. Toy Corpus & Vocabulary
# -------------------------------
corpus = [
    "the king loves the queen",
    "the queen loves the king",
    "the king and the queen love the crown",
    "apple and fruit are tasty"
]

# Tokenize
tokenized = [sentence.lower().split() for sentence in corpus]
vocab = sorted(set(word for sent in tokenized for word in sent))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}
V = len(vocab)
print("Vocabulary:", vocab)

# -------------------------------
# 2. Build Co-occurrence Matrix
# -------------------------------
X = np.zeros((V, V))
window_size = 2

for sentence in tokenized:
    for i, word in enumerate(sentence):
        w_id = word2id[word]
        for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
            if i != j:
                context_id = word2id[sentence[j]]
                X[w_id, context_id] += 1

print("\nCo-occurrence matrix:\n", X)

# -------------------------------
# 3. Initialize GloVe Parameters
# -------------------------------
embedding_dim = 5
np.random.seed(42)

W = np.random.randn(V, embedding_dim)     # word vectors
W_tilde = np.random.randn(V, embedding_dim)  # context vectors
b = np.zeros(V)       # biases
b_tilde = np.zeros(V)

# Weighting function
def f(x, xmax=100, alpha=0.75):
    return (x / xmax) ** alpha if x < xmax else 1

fX = np.vectorize(f)(X)

# -------------------------------
# 4. Training Loop
# -------------------------------
epochs = 200
lr = 0.05

for epoch in range(epochs):
    loss = 0
    for i in range(V):
        for j in range(V):
            if X[i, j] > 0:
                w_i = W[i]
                w_j = W_tilde[j]
                dot = np.dot(w_i, w_j) + b[i] + b_tilde[j]
                cost = dot - np.log(X[i, j])
                weight = fX[i, j]
                loss += 0.5 * weight * (cost ** 2)

                # Gradients
                grad = weight * cost
                W[i]       -= lr * grad * w_j
                W_tilde[j] -= lr * grad * w_i
                b[i]       -= lr * grad
                b_tilde[j] -= lr * grad
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -------------------------------
# 5. Final Embeddings
# -------------------------------
embeddings = W + W_tilde

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\nSimilarity(king, queen) =", cosine_similarity(embeddings[word2id["king"]], embeddings[word2id["queen"]]))
print("Similarity(apple, fruit) =", cosine_similarity(embeddings[word2id["apple"]], embeddings[word2id["fruit"]]))

# -------------------------------
# 6. Visualization in 2D
# -------------------------------
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i, word in id2word.items():
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x+0.02, y+0.02, word, fontsize=12)
plt.title("GloVe Embeddings (Toy Corpus)")
plt.show()
