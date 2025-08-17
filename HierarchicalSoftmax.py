import numpy as np

# Toy embeddings
emb_dim = 5
vocab = ["king", "queen", "apple", "fruit"]
word_to_code = {
    "king":  [(0, "root")],                   # left from root
    "queen": [(1, "root"), (0, "node1")],     # right->left
    "apple": [(1, "root"), (1, "node1"), (0, "node2")],
    "fruit": [(1, "root"), (1, "node1"), (1, "node2")]
}

# Parameters for internal nodes
internal_nodes = {"root": np.random.randn(emb_dim),
                  "node1": np.random.randn(emb_dim),
                  "node2": np.random.randn(emb_dim)}

embeddings = {w: np.random.randn(emb_dim) for w in vocab}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hierarchical_softmax_prob(word, context):
    v_c = embeddings[context]
    code = word_to_code[word]
    prob = 1.0
    for direction, node in code:
        score = np.dot(internal_nodes[node], v_c)
        p = sigmoid(score)
        prob *= (p if direction == 1 else (1 - p))
    return prob

# Example: probability of "king" given context "queen"
p = hierarchical_softmax_prob("king", "queen")
print("P(king | queen) =", p)
