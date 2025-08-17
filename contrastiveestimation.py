import numpy as np

# toy embeddings
vocab = ["king", "queen", "apple", "fruit"]
emb_dim = 5
embeddings = {w: np.random.randn(emb_dim) for w in vocab}

def score(word, context):
    """dot product score between word and context"""
    return np.dot(embeddings[word], embeddings[context])

def contrastive_loss(word, pos_context, neg_contexts, margin=1.0):
    """Skip-gram style contrastive estimation"""
    pos_score = score(word, pos_context)
    neg_scores = [score(word, neg) for neg in neg_contexts]
    # margin hinge loss
    losses = [max(0, neg + margin - pos_score) for neg in neg_scores]
    return pos_score, neg_scores, sum(losses)

# Example: word = king, positive context = queen, negatives = apple, fruit
word = "king"
pos_context = "queen"
neg_contexts = ["apple", "fruit"]

pos_score, neg_scores, loss = contrastive_loss(word, pos_context, neg_contexts)

print("Positive score:", pos_score)
print("Negative scores:", neg_scores)
print("Loss:", loss)
