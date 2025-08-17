import numpy as np

# Corpus
corpus = "the cat sits on the mat".split()

# Vocabulary
vocab = list(set(corpus))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)

# One-hot (not really used directly here)
def one_hot(word):
    vec = np.zeros(V)
    vec[word2idx[word]] = 1
    return vec

# Training data (center → context pairs)
window_size = 2
training_data = []
for i, word in enumerate(corpus):
    for j in range(-window_size, window_size+1):
        if j != 0 and 0 <= i+j < len(corpus):
            training_data.append((word, corpus[i+j]))

print("Training pairs (Skip-Gram):")
print(training_data[:6])

# Model parameters
embedding_dim = 5
W_in = np.random.randn(V, embedding_dim) * 0.01   # input (center)
W_out = np.random.randn(embedding_dim, V) * 0.01 # output (context)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Negative sampling
def get_negative_samples(true_idx, num_neg=3):
    negs = []
    while len(negs) < num_neg:
        neg = np.random.randint(0, V)
        if neg != true_idx:  # avoid picking the true context
            negs.append(neg)
    return negs

# Training
lr = 0.05
num_neg_samples = 3
for epoch in range(500):
    loss = 0
    for center, context in training_data:
        center_idx = word2idx[center]
        context_idx = word2idx[context]

        # Embeddings
        v_c = W_in[center_idx]   # center word vector
        u_o = W_out[:, context_idx]  # context word vector

        # Positive sample loss (center, context)
        score_pos = sigmoid(np.dot(v_c, u_o))
        loss += -np.log(score_pos + 1e-9)

        # Gradients for positive sample
        grad_pos = (score_pos - 1)
        W_in[center_idx] -= lr * grad_pos * u_o
        W_out[:, context_idx] -= lr * grad_pos * v_c

        # Negative samples
        neg_samples = get_negative_samples(context_idx, num_neg_samples)
        for neg_idx in neg_samples:
            u_neg = W_out[:, neg_idx]
            score_neg = sigmoid(np.dot(v_c, u_neg))
            loss += -np.log(1 - score_neg + 1e-9)

            # Gradients for negative sample
            grad_neg = score_neg
            W_in[center_idx] -= lr * grad_neg * u_neg
            W_out[:, neg_idx] -= lr * grad_neg * v_c

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final embeddings
print("\nLearned word embeddings (SGNS):")
for word in vocab:
    print(word, "→", W_in[word2idx[word]])
