import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function and its gradient
def f(w):
    return (w - 3)**2

def grad_f(w):
    return 2 * (w - 3)

# Batch Gradient Descent (Vanilla)
def batch_gd(initial_w, lr=0.1, iterations=50):
    w = initial_w
    w_history = [w]
    for _ in range(iterations):
        w = w - lr * grad_f(w)
        w_history.append(w)
    return w_history

# Momentum Gradient Descent
def momentum_gd(initial_w, lr=0.1, beta=0.9, iterations=50):
    w = initial_w
    u = 0
    w_history = [w]
    for _ in range(iterations):
        u = beta * u + grad_f(w)
        w = w - lr * u
        w_history.append(w)
    return w_history

# Nesterov Accelerated Gradient
def nag_gd(initial_w, lr=0.1, beta=0.9, iterations=50):
    w = initial_w
    u = 0
    w_history = [w]
    for _ in range(iterations):
        lookahead = w - lr * beta * u
        grad = grad_f(lookahead)
        u = beta * u + grad
        w = w - lr * u
        w_history.append(w)
    return w_history

# Stochastic Gradient Descent (simulated by adding noise to gradient)
def stochastic_gd(initial_w, lr=0.1, noise_scale=0.1, iterations=50):
    w = initial_w
    w_history = [w]
    for _ in range(iterations):
        noise = np.random.normal(0, noise_scale)
        grad = grad_f(w) + noise
        w = w - lr * grad
        w_history.append(w)
    return w_history

# Run all optimizers
batch_path = batch_gd(0.0)
momentum_path = momentum_gd(0.0)
nag_path = nag_gd(0.0)
stochastic_path = stochastic_gd(0.0)

# Ensure all paths are same length for animation
min_len = min(len(batch_path), len(momentum_path), len(nag_path), len(stochastic_path))

# Plot setup
w_vals = np.linspace(-1, 6, 200)
f_vals = f(w_vals)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(w_vals, f_vals, 'k-', label='f(w) = (w - 3)^2')
ax.set_xlim(-1, 6)
ax.set_ylim(0, 10)
ax.set_xlabel("w")
ax.set_ylabel("f(w)")
ax.set_title("Batch vs Momentum vs NAG vs Stochastic Gradient Descent")

# Add animated dots
batch_dot,     = ax.plot([], [], 'bo', label='Batch GD', markersize=12, alpha=0.9)
momentum_dot,  = ax.plot([], [], 'ro', label='Momentum GD')
nag_dot,       = ax.plot([], [], 'go', label='NAG')
sgd_dot,       = ax.plot([], [], 'mo', label='Stochastic GD')

ax.legend()

# Initialization
def init():
    batch_dot.set_data([], [])
    momentum_dot.set_data([], [])
    nag_dot.set_data([], [])
    sgd_dot.set_data([], [])
    return batch_dot, momentum_dot, nag_dot, sgd_dot

# Update function
def update(i):
    b_w = batch_path[i]
    m_w = momentum_path[i]
    n_w = nag_path[i]
    s_w = stochastic_path[i]
    batch_dot.set_data([b_w], [f(b_w)])
    momentum_dot.set_data([m_w], [f(m_w)])
    nag_dot.set_data([n_w], [f(n_w)])
    sgd_dot.set_data([s_w], [f(s_w)])
    return batch_dot, momentum_dot, nag_dot, sgd_dot

# Create animation
ani = FuncAnimation(fig, update, frames=min_len, init_func=init, blit=True)

plt.show()

# Optional save
# ani.save("gradient_descent_all.gif", writer="pillow")
