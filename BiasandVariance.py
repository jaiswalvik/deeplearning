import numpy as np
import sympy as sp

# Step 1: Define symbolic variables
w, b = sp.symbols('w b')

# Step 2: Define the loss function L(w, b)
L = 0.5 * w**2 + 5 * b**2 + 1

# Step 3: Compute the gradient ∇L
grad_L = [sp.diff(L, var) for var in (w, b)]
print("Gradient ∇L(w, b):", grad_L)

# Step 4: Compute the Hessian matrix H_L
Hessian = sp.Matrix([[sp.diff(grad_L[i], var) for var in (w, b)] for i in range(2)])
print("\nHessian H_L(w, b):")
sp.pprint(Hessian)

# Step 5: Evaluate Hessian at (w=0, b=0) and convert to NumPy
H = np.array(Hessian.subs({w: 0, b: 0})).astype(np.float64)
print("\nNumerical Hessian at (0, 0):\n", H)

# Step 6: Eigenvalue decomposition of the Hessian
eigvals, eigvecs = np.linalg.eigh(H)
print("\nEigenvalues of Hessian:", eigvals)
print("Eigenvectors of Hessian:\n", eigvecs)

# Step 7: Apply L2 regularization
lambda_reg = 2.0  # Regularization parameter
H_reg = H + 2 * lambda_reg * np.eye(2)
print("\nRegularized Hessian (H + 2λI):\n", H_reg)

# Step 8: Verify minimum remains at (w, b) = (0, 0)
w_star = 0
b_star = 0
L_reg = (0.5 + lambda_reg)*w_star**2 + (5 + lambda_reg)*b_star**2 + 1
print("\nRegularized loss at (0, 0):", L_reg)
print("Minimum still at (w, b) = (0, 0)?", (w_star == 0 and b_star == 0))
