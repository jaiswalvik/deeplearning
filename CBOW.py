import numpy as np

# Corpus
corpus = "the cat sits on the mat".split()

# Vocabulary
vocab = list(set(corpus))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)  # vocab size

# One-hot encoding function
def one_hot(word):
    vec = np.zeros(V)
    vec[word2idx[word]] = 1
    return vec

# Training data preparation (context -> target)
window_size = 2
training_data = []
for i, word in enumerate(corpus):
    context = []
    for j in range(-window_size, window_size+1):
        if j != 0 and 0 <= i+j < len(corpus):
            context.append(corpus[i+j])
    if context:
        training_data.append((context, word))

print("Training samples:")
for ctx, tgt in training_data:
    print(f"Context: {ctx} → Target: {tgt}")

# Build CBOW model parameters
embedding_dim = 5
W1 = np.random.randn(V, embedding_dim) * 0.01   # input → hidden
W2 = np.random.randn(embedding_dim, V) * 0.01   # hidden → output

# Softmax
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Training
lr = 0.1
for epoch in range(500):
    loss = 0
    for context_words, target in training_data:
        # Step 1: Input
        x = np.mean([one_hot(w) for w in context_words], axis=0)  # average of context
        
        # Step 2: Forward pass
        h = np.dot(x, W1)        # hidden
        u = np.dot(h, W2)        # output layer scores
        y_pred = softmax(u)      # prediction

        # Step 3: Loss (cross-entropy)
        y_true = one_hot(target)
        loss -= np.log(y_pred[np.argmax(y_true)] + 1e-9)

        # Step 4: Backpropagation
        error = y_pred - y_true
        dW2 = np.outer(h, error)
        dW1 = np.outer(x, np.dot(W2, error))

        # Step 5: Update weights
        W1 -= lr * dW1
        W2 -= lr * dW2
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Show learned embeddings
print("\nWord embeddings:")
for word in vocab:
    print(word, "→", W1[word2idx[word]])
