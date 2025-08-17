A collection of course assignments and python code for studying Deep Learning taught at IIT Madras BS Data Science and Applications:

https://www.youtube.com/playlist?list=PLZ2ps__7DhBZVxMrSkTIcG6zZBDKUXCnM

Notes: 
# Word Embedding Formulas

## 1. SVD with Co-occurrence Matrix
We start with a **word-word co-occurrence matrix** \(X \in \mathbb{R}^{V \times V}\), where \(X_{ij}\) = number of times word \(i\) co-occurs with word \(j\).

**Singular Value Decomposition (SVD):**
\[
X = U \Sigma V^T
\]

Keep top-\(k\) dimensions:
\[
W = U_k \Sigma_k^{1/2}, \quad C = V_k \Sigma_k^{1/2}
\]

Final word embeddings ≈ rows of \(W\).

---

## 2. Continuous Bag-of-Words (CBOW)
Given context words \((w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m})\), predict center word \(w_t\).

**Context vector:**
\[
h = \frac{1}{2m} \sum_{-m \leq j \leq m, j \neq 0} v_{w_{t+j}}
\]

**Prediction (softmax):**
\[
P(w_t \,|\, context) = \frac{\exp(u_{w_t}^T h)}{\sum_{w \in V} \exp(u_{w}^T h)}
\]

**Loss:**
\[
J = - \log P(w_t \,|\, context)
\]

---

## 3. Skip-Gram
Given center word \(w_t\), predict surrounding context words.

**Probability of context word:**
\[
P(w_{t+j} \,|\, w_t) = \frac{\exp(u_{w_{t+j}}^T v_{w_t})}{\sum_{w \in V} \exp(u_w^T v_{w_t})}
\]

**Objective:**
\[
J = \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} \,|\, w_t)
\]

---

## 4. GloVe (Global Vectors)
Uses co-occurrence statistics \(X_{ij}\).

**Objective:**
\[
J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2
\]

Where:
- \(f(x) = \min\big((x/x_{\max})^\alpha, 1\big)\)  
- \(w_i\) = word vector, \(\tilde{w}_j\) = context vector  

---

## 5. Hierarchical Softmax
Softmax replaced with binary tree (Huffman coding).

**Probability of word \(w\) given context \(h\):**
\[
P(w \,|\, h) = \prod_{j=1}^{L(w)} \sigma \left( \llbracket n_j(w) = \text{left} \rrbracket \cdot u_{n_j}^T h \right)
\]

Where:
- \(L(w)\) = path length to leaf node of \(w\)  
- \(n_j(w)\) = \(j\)-th node along path  
- \(\sigma(x) = \frac{1}{1 + e^{-x}}\)  
- Indicator = \(+1\) if branch is left, \(-1\) if right  

---

## 6. Contrastive Estimation (Negative Sampling)
Approximation to softmax. Contrast positive pairs with sampled negatives.

**Objective (for center word \(w_t\), context word \(w_c\)):**
\[
J = \log \sigma(u_{w_c}^T v_{w_t}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-u_{w_i}^T v_{w_t}) \right]
\]

Where:
- \(k\) = number of negative samples  
- \(P_n(w)\) = noise distribution  
