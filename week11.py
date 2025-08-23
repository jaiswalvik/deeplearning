import numpy as np
from pathlib import Path

PARAMS_IN  = Path("parameters_w11.npz")  # optional parameter file

# -----------------------------
# Utilities
# -----------------------------
def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx, 0] = 1.0
    return v

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

# -----------------------------
# Data (unique source characters, target chars)
# -----------------------------
src_vocab = ['a','r','i','y']      # unique source chars
tgt_vocab = ['l','e','a','r','n']  # target 5 chars

src_index = {c:i for i,c in enumerate(src_vocab)}
tgt_index = {c:i for i,c in enumerate(tgt_vocab)}
tgt_from_idx = {i:c for c,i in tgt_index.items()}

source_text = "ariya"
target_text = "learn"

# Encoder input sequence
source_seq = [one_hot(src_index[c], 5) for c in source_text]  # 5 steps
# Decoder inputs for teacher forcing: <GO> is zero vector
decoder_inputs = [np.zeros((5,1))] + [one_hot(tgt_index[c], 5) for c in target_text[:-1]]
target_indices = [tgt_index[c] for c in target_text]

# -----------------------------
# Load / initialize parameters
# -----------------------------
if PARAMS_IN.exists():
    p = np.load(PARAMS_IN)
    U_e, W_e, W_d, U_d, V_d = [p[k].astype(np.float64) for k in ['U_e','W_e','W_d','U_d','V_d']]
    print("\n--- Loaded Parameters ---")
else:
    rng = np.random.default_rng(0)
    U_e = rng.standard_normal((5,5))*0.1
    W_e = rng.standard_normal((5,5))*0.1
    W_d = rng.standard_normal((5,5))*0.1
    U_d = rng.standard_normal((5,5))*0.1
    V_d = rng.standard_normal((5,5))*0.1

# -----------------------------
# Forward, Loss, BPTT
# -----------------------------
def forward(U_e, W_e, W_d, U_d, V_d):
    h = [np.zeros((5,1))]   # encoder hidden states
    a_enc = []
    for x in source_seq:
        a = W_e @ h[-1] + U_e @ x
        a_enc.append(a)
        h.append(np.tanh(a))
    
    s = [h[-1]]   # decoder hidden states
    a_dec = []
    y_list = []
    for z in decoder_inputs:
        a = W_d @ s[-1] + U_d @ z
        a_dec.append(a)
        st = np.tanh(a)
        s.append(st)
        y_list.append(softmax(V_d @ st))
    return h, a_enc, s, a_dec, y_list

def loss_from_preds(y_list):
    loss = 0.0
    for t, y in enumerate(y_list):
        loss -= np.log(y[target_indices[t], 0] + 1e-9)
    return loss

def bptt(U_e, W_e, W_d, U_d, V_d, h, a_enc, s, a_dec, y_list):
    gU_e = np.zeros_like(U_e)
    gW_e = np.zeros_like(W_e)
    gW_d = np.zeros_like(W_d)
    gU_d = np.zeros_like(U_d)
    gV_d = np.zeros_like(V_d)

    g_s_next = np.zeros((5,1))
    T = len(y_list)
    for t in reversed(range(T)):
        y = y_list[t].copy()
        y[target_indices[t],0] -= 1.0
        gV_d += y @ s[t+1].T
        g_s = V_d.T @ y + g_s_next
        da = (1 - np.tanh(a_dec[t])**2) * g_s
        gW_d += da @ s[t].T
        gU_d += da @ decoder_inputs[t].T
        g_s_next = W_d.T @ da

    g_h_next = g_s_next
    for t in reversed(range(len(source_seq))):
        da = (1 - np.tanh(a_enc[t])**2) * g_h_next
        gW_e += da @ h[t].T
        gU_e += da @ source_seq[t].T
        g_h_next = W_e.T @ da

    return gU_e, gW_e, gW_d, gU_d, gV_d

# -----------------------------
# Decoding
# -----------------------------
def decode_greedy(y_list):
    return ''.join(tgt_from_idx[int(np.argmax(y))] for y in y_list)

def decode_autoregressive(U_e, W_e, W_d, U_d, V_d, max_len=5):
    h = [np.zeros((5,1))]
    for x in source_seq:
        h.append(np.tanh(W_e @ h[-1] + U_e @ x))
    s = [h[-1]]
    preds = []
    z = np.zeros((5,1))
    for t in range(max_len):
        a = W_d @ s[-1] + U_d @ z
        st = np.tanh(a)
        s.append(st)
        y = softmax(V_d @ st)
        idx = int(np.argmax(y))
        preds.append(tgt_from_idx[idx])
        z = one_hot(idx, 5)
    return ''.join(preds)

# -----------------------------
# Training
# -----------------------------
eta = 1.0
epochs = 20
clip = 5.0

for ep in range(1, epochs+1):
    h, a_enc, s, a_dec, y_list = forward(U_e, W_e, W_d, U_d, V_d)
    L = loss_from_preds(y_list)
    auto_pred = decode_autoregressive(U_e, W_e, W_d, U_d, V_d)
    gU_e, gW_e, gW_d, gU_d, gV_d = bptt(U_e, W_e, W_d, U_d, V_d, h, a_enc, s, a_dec, y_list)

    # gradient clipping
    for g in (gU_e, gW_e, gW_d, gU_d, gV_d):
        np.clip(g, -clip, clip, out=g)

    # SGD update
    U_e -= eta * gU_e
    W_e -= eta * gW_e
    W_d -= eta * gW_d
    U_d -= eta * gU_d
    V_d -= eta * gV_d

    print(f"Epoch {ep:4d} | loss: {L:.4f} | pred: {decode_greedy(y_list)} | autoreg: {auto_pred}")

# -----------------------------
# Final report
# -----------------------------
h, a_enc, s, a_dec, y_list = forward(U_e, W_e, W_d, U_d, V_d)
final_pred = decode_greedy(y_list)
final_loss = loss_from_preds(y_list)

print("\nFinal loss:", final_loss)
print("Final prediction:", final_pred)


