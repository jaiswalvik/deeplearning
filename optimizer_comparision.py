import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Objective function and gradient
def f(w):
    x, y = w
    return (x - 3) ** 2 + (y + 1) ** 2

def grad_f(w):
    x, y = w
    return np.array([2 * (x - 3), 2 * (y + 1)])

# Optimizers
def adagrad_path(grad_f, w0, lr=0.1, eps=1e-8, steps=300):
    w, G = w0.copy(), np.zeros_like(w0)
    path = [w.copy()]
    for _ in range(steps):
        g = grad_f(w)
        G += g ** 2
        w -= lr * g / (np.sqrt(G) + eps)
        path.append(w.copy())
    return np.array(path)

def rmsprop_path(grad_f, w0, lr=0.02, beta=0.9, eps=1e-8, steps=300):
    w, Eg = w0.copy(), np.zeros_like(w0)
    path = [w.copy()]
    for _ in range(steps):
        g = grad_f(w)
        Eg = beta * Eg + (1 - beta) * g ** 2
        w -= lr * g / (np.sqrt(Eg) + eps)
        path.append(w.copy())
    return np.array(path)

def adadelta_path(grad_f, w0, rho=0.95, eps=1e-6, steps=300):
    w = w0.copy()
    Eg, Edx = np.zeros_like(w0), np.zeros_like(w0)
    path = [w.copy()]
    for _ in range(steps):
        g = grad_f(w)
        Eg = rho * Eg + (1 - rho) * g ** 2
        dx = - (np.sqrt(Edx + eps) / np.sqrt(Eg + eps)) * g
        w += dx
        Edx = rho * Edx + (1 - rho) * dx ** 2
        path.append(w.copy())
    return np.array(path)

def adam_path(grad_f, w0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, steps=300):
    w, m, v = w0.copy(), np.zeros_like(w0), np.zeros_like(w0)
    path = [w.copy()]
    for t in range(1, steps + 1):
        g = grad_f(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(w.copy())
    return np.array(path)

def adamax_path(grad_f, w0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, steps=300):
    w, m, u = w0.copy(), np.zeros_like(w0), np.zeros_like(w0)
    path = [w.copy()]
    for t in range(1, steps + 1):
        g = grad_f(w)
        m = beta1 * m + (1 - beta1) * g
        u = np.maximum(beta2 * u, np.abs(g))
        m_hat = m / (1 - beta1 ** t)
        w -= lr * m_hat / (u + eps)
        path.append(w.copy())
    return np.array(path)

def nadam_path(grad_f, w0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, steps=300):
    w, m, v = w0.copy(), np.zeros_like(w0), np.zeros_like(w0)
    path = [w.copy()]
    for t in range(1, steps + 1):
        g = grad_f(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        nesterov = beta1 * m_hat + (1 - beta1) * g / (1 - beta1 ** t)
        w -= lr * nesterov / (np.sqrt(v_hat) + eps)
        path.append(w.copy())
    return np.array(path)

# Convergence detection
def has_converged(path, tol=1e-3):
    target = np.array([3.0, -1.0])
    distances = np.linalg.norm(path - target, axis=1)
    for i, d in enumerate(distances):
        if d < tol:
            return i
    return len(path)

# Initial point and max possible steps
initial_w = np.array([5.0, 5.0])
max_possible_steps = 1000

# Temporary long paths to check convergence
temp_paths = {
    "Adagrad": adagrad_path(grad_f, initial_w, steps=max_possible_steps),
    "RMSProp": rmsprop_path(grad_f, initial_w, steps=max_possible_steps),
    "AdaDelta": adadelta_path(grad_f, initial_w, steps=max_possible_steps),
    "Adam": adam_path(grad_f, initial_w, steps=max_possible_steps),
    "AdaMax": adamax_path(grad_f, initial_w, steps=max_possible_steps),
    "Nadam": nadam_path(grad_f, initial_w, steps=max_possible_steps)
}

# Determine how many steps are needed for each
convergence_steps = {
    name: has_converged(path)
    for name, path in temp_paths.items()
}
steps = max(convergence_steps.values())
print("Convergence steps:", convergence_steps)
print("Max steps for animation:", steps)

# Actual paths truncated to convergence
paths = {
    "Adagrad": adagrad_path(grad_f, initial_w, steps=steps),
    "RMSProp": rmsprop_path(grad_f, initial_w, steps=steps),
    "AdaDelta": adadelta_path(grad_f, initial_w, steps=steps),
    "Adam": adam_path(grad_f, initial_w, steps=steps),
    "AdaMax": adamax_path(grad_f, initial_w, steps=steps),
    "Nadam": nadam_path(grad_f, initial_w, steps=steps)
}

# Colors
colors = {
    "Adagrad": "red",
    "RMSProp": "green",
    "AdaDelta": "blue",
    "Adam": "magenta",
    "AdaMax": "orange",
    "Nadam": "cyan"
}

# Contour plot
x = np.linspace(-1, 7, 100)
y = np.linspace(-5, 6, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

fig, ax = plt.subplots(figsize=(10, 7))
ax.contour(X, Y, Z, levels=30, cmap='viridis')
ax.plot(3, -1, 'k*', markersize=12, label='Minimum')
ax.set_xlim(-1, 7)
ax.set_ylim(-5, 6)
ax.set_title("Adaptive Optimizer Paths on Contour Plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Initialize animated lines
lines = {
    name: ax.plot([], [], '-', lw=1, color=colors[name], label=name)[0]
    for name in paths
}
ax.legend()

# Animation function
def animate(i):
    for name, path in paths.items():
        segment = path[:i + 1]
        lines[name].set_data(segment[:, 0], segment[:, 1])
    return list(lines.values())

ani = FuncAnimation(fig, animate, frames=steps + 1, interval=50, blit=False)
plt.show()
