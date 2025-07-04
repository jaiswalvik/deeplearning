import numpy as np
import matplotlib.pyplot as plt

# Common settings
eta = 0.1
iterations = 10
initial_w = -2

# --------- Adam Optimizer ---------
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
w_adam = initial_w
m = 0
v = 0
adam_weights = [w_adam]

# --------- Momentum GD ---------
beta = 0.9
w_momentum = initial_w
velocity = 0
momentum_weights = [w_momentum]

# Gradient of L(w) = w^2
def gradient(w):
    return 2 * w

for t in range(1, iterations + 1):
    # ---- Adam ----
    g = gradient(w_adam)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w_adam = w_adam - eta * m_hat / (np.sqrt(v_hat) + epsilon)
    print(f"Adam iteration {t}: w = {w_adam}, m = {m}, v = {v}")
    adam_weights.append(w_adam)

    # ---- Momentum GD ----
    g_mgd = gradient(w_momentum)
    velocity = beta * velocity + eta * g_mgd
    w_momentum = w_momentum - velocity
    print(f"Momentum GD iteration {t}: w = {w_momentum}, velocity = {velocity}")
    momentum_weights.append(w_momentum)

# -------- Plotting ----------
plt.figure(figsize=(10, 5))
plt.plot(range(iterations + 1), adam_weights, marker='o', label="Adam")
plt.plot(range(iterations + 1), momentum_weights, marker='s', label="Momentum GD")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("Weight (w)")
plt.title("Adam vs Momentum-based Gradient Descent on L(w) = w²")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
