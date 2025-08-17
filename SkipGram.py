import numpy as np

# Corpus
corpus = "the cat sits on the mat".split()

# Vocabulary
vocab = list(set(corpus))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)

# One-hot encoding
def one_hot(word):
    vec = np.zeros(V)
    vec[word2idx[word]] = 1
    return vec

# Training data (center → context)
window_size = 2
training_data = []
for i, word in enumerate(corpus):
    for j in range(-window_size, window_size+1):
        if j != 0 and 0 <= i+j < len(corpus):
            training_data.append((word, corpus[i+j]))

print("Training samples (Skip-Gram):")
for c, ctx in training_data[:6]:
    print(f"Center: {c} → Context: {ctx}")

# Model parameters
embedding_dim = 5
W1 = np.random.randn(V, embedding_dim) * 0.01
W2 = np.random.randn(embedding_dim, V) * 0.01

# Softmax
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Training
lr = 0.1
for epoch in range(500):
    loss = 0
    for center, context in training_data:
        x = one_hot(center)
        
        # Forward
        h = np.dot(x, W1)   # embedding of center
        u = np.dot(h, W2)
        y_pred = softmax(u)

        # True label
        y_true = one_hot(context)
        loss -= np.log(y_pred[np.argmax(y_true)] + 1e-9)

        # Backprop
        error = y_pred - y_true
        dW2 = np.outer(h, error)
        dW1 = np.outer(x, np.dot(W2, error))

        W1 -= lr * dW1
        W2 -= lr * dW2
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Word embeddings
print("\nWord embeddings (Skip-Gram):")
for word in vocab:
    print(word, "→", W1[word2idx[word]])
