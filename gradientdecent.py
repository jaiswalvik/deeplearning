import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Dataset
X = [-1, 0.5]
Y = [0.5, 0.03]

# Sigmoid function
def f(x, w, b):
    return 1 / (1 + np.exp(-(w * x - b)))

# Error function
def error(w, b):
    return 0.5 * sum((f(x, w, b) - y) ** 2 for x, y in zip(X, Y))

# Gradients
def grad_w(x, w, b, y):
    fx = f(x, w, b)
    return (fx - y) * fx * (1 - fx) * x

def grad_b(x, w, b, y):
    fx = f(x, w, b)
    return (fx - y) * fx * (1 - fx)

# Track history
w_history = []
b_history = []
err_history = []

def do_gradient_descent():
    w, b = 2, 2
    eta = 1.0
    max_epochs = 42
    for _ in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)
        w -= eta * dw
        b -= eta * db
        w_history.append(w)
        b_history.append(b)
        err_history.append(error(w, b))

do_gradient_descent()

# Setup figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, len(err_history))
ax.set_ylim(0, max(err_history) * 1.1)
line, = ax.plot([], [], 'r-', label='Error')
text_box = ax.text(0.7, 0.95, '', transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='black'))
ax.set_title("Gradient Descent Error Convergence")
ax.set_xlabel("Epoch")
ax.set_ylabel("Error")
ax.grid(True)
ax.legend()

# Init function
def init():
    line.set_data([], [])
    text_box.set_text('')
    return line, text_box

# Update function (shows w, b changing each frame)
def update(frame):
    x_data = list(range(frame + 1))
    y_data = err_history[:frame + 1]
    line.set_data(x_data, y_data)

    current_w = w_history[frame]
    current_b = b_history[frame]
    text_box.set_text(f'Epoch {frame+1}\nw={current_w:.4f}\nb={current_b:.4f}')
    
    return line, text_box

ani = animation.FuncAnimation(
    fig, update, frames=len(err_history), init_func=init,
    interval=100, blit=True, repeat=False
)

plt.show()
