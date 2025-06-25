import numpy as np

# Define the function and its gradient
def f(w):
    return 0.1*(w**2)

def grad_f(w):
    return 0.2 * w

# Momentum Gradient Descent
def momentum_gd(initial_w, lr=10.0, beta=0.9, iterations=100):
    w = initial_w
    u = 0
    i=0
    for _ in range(iterations):
        u = beta * u + grad_f(w)
        w = w - lr * u
        print(f"Iteration:",i)
        print(f"w:",w)
        print(f"Loss:",f(w))
        i+=1
    return w

print(f"after 100 epochs loss:",f(momentum_gd(1.0)))