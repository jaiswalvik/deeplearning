import numpy as np
import matplotlib.pyplot as plt

# Corpus
docs = [
    "knowing the name of something is different from knowing something",
    "knowing something about everything is alright"
]

# Step 1: Tokenize & lowercase
tokenized_docs = [doc.lower().split() for doc in docs]

# Step 2: Build vocabulary (unique words)
vocab = sorted(set(word for doc in tokenized_docs for word in doc))
print("Vocabulary:", vocab)
print("V =", len(vocab))

# Words to drop
drop_words = {"of", "the", "alright", "about", "from"}

# Step 3: Tokenize, lowercase, and filter unwanted words
filtered_docs = [
    [w for w in doc.lower().split() if w not in drop_words]
    for doc in docs
]

# Step 4: Build vocabulary from filtered corpus
vocab = sorted(set(word for doc in filtered_docs for word in doc))
word_to_idx = {word: i for i, word in enumerate(vocab)}
V = len(vocab)

print("Vocabulary:", vocab)

# Step 5: Initialize co-occurrence matrix
cooc = np.zeros((V, V), dtype=int)

# Step 6: Fill co-occurrence matrix (window size k=1) - no double counting
window_size = 1
N = 0
for sent in filtered_docs:
    for i, word in enumerate(sent):
        if word not in word_to_idx:
            continue
        w_idx = word_to_idx[word]
        # Look only right neighbors to avoid double counting
        for j in range(1, window_size + 1):
            if i + j < len(sent):
                neighbor = sent[i + j]
                if neighbor in word_to_idx:
                    n_idx = word_to_idx[neighbor]
                    cooc[w_idx, n_idx] += 1
                    cooc[n_idx, w_idx] += 1  # enforce symmetry immediately
                    N += 1
# Step 7: Results
print("Total co-occurrences (N):", N)
print("\nCo-occurrence matrix:\n", cooc)
nonzero_count = np.count_nonzero(cooc)
print("\nNon-zero entries:", nonzero_count)

# Step 8: Heatmap
plt.figure(figsize=(6,5))
plt.imshow(cooc, cmap='Blues', interpolation='nearest')
plt.colorbar(label="Co-occurrence Count")
plt.xticks(range(V), vocab, rotation=45)
plt.yticks(range(V), vocab)
plt.title("Co-occurrence Matrix Heatmap (k=1, symmetric, no double count)")
plt.tight_layout()
plt.show()

# Normalize each row to unit length
norms = np.linalg.norm(cooc, axis=1, keepdims=True)
norms[norms == 0] = 1
unit_vectors = cooc / norms

# Cosine similarity = dot product of normalized vectors
cos_sim_matrix = np.dot(unit_vectors, unit_vectors.T)

# Find closest word to "knowing"
knowing_idx = vocab.index("knowing")
similarities = cos_sim_matrix[knowing_idx]

# Ignore self-similarity by setting it to -1
similarities[knowing_idx] = -1
closest_idx = np.argmax(similarities)

print("Similaritites to 'knowing':", similarities)
print("Closest word to 'knowing':", vocab[closest_idx])
print("Cosine similarity:", similarities[closest_idx])


# Compute full SVD
U, S, VT = np.linalg.svd(unit_vectors, full_matrices=False)

def rank_k_approximation(U, S, VT, k):
    """Return rank-k approximation matrix."""
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    VTk = VT[:k, :]
    return Uk @ Sk @ VTk

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vocab = ['different', 'everything', 'is', 'knowing', 'name', 'something']
knowing_idx = vocab.index('knowing')

for rank in [1, 2, 3]:
    print(f"\nRank-{rank} approximation:")

    # Get rank-k approx matrix
    M_k = rank_k_approximation(U, S, VT, rank)

    # Vector for "knowing"
    knowing_vec = M_k[knowing_idx, :]

    # Compute similarities
    similarities = []
    for i in range(len(vocab)):
        sim = cosine_sim(knowing_vec, M_k[i, :])
        similarities.append(sim)

    # Print words with similarity > 0.5 excluding "knowing"
    for i, sim in enumerate(similarities):
        if i != knowing_idx and sim > 0.5:
            print(f"{vocab[i]}: similarity = {sim:.3f}")
