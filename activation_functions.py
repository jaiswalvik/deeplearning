import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Activation Functions ---
def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# --- Derivatives ---
def d_relu(x):
    return (x > 0).astype(float)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def d_silu(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def d_gelu(x):
    tanh_out = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
    left = 0.5 * tanh_out
    right = (0.5 * x * (1 - tanh_out ** 2)) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
    return left + right + 0.5

# --- Input Range ---
x = np.linspace(-6, 6, 1000)

# --- Compute Outputs ---
activations = {
    "ReLU": relu(x),
    "Tanh": tanh(x),
    "Sigmoid": sigmoid(x),
    "SiLU (Swish)": silu(x),
    "GELU": gelu(x)
}

derivatives = {
    "ReLU": d_relu(x),
    "Tanh": d_tanh(x),
    "Sigmoid": d_sigmoid(x),
    "SiLU (Swish)": d_silu(x),
    "GELU": d_gelu(x)
}

# --- Plot Activation Functions ---
plt.figure(figsize=(12, 6))
for name, y in activations.items():
    plt.plot(x, y, label=name)
plt.title("Activation Functions")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()

# --- Plot Derivatives ---
plt.figure(figsize=(12, 6))
for name, y in derivatives.items():
    plt.plot(x, y, label=f"{name}'")
plt.title("Derivatives of Activation Functions")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()

# --- Comparison Table ---
data = {
    "Function": ["ReLU", "Tanh", "Sigmoid", "SiLU (Swish)", "GELU"],
    "Smooth": ["No", "Yes", "Yes", "Yes", "Yes"],
    "Monotonic": ["Yes", "Yes", "Yes", "No", "No"],
    "Negative Output": ["No", "Yes", "Yes", "Yes", "Yes"],
    "Used in": ["CNNs", "Old NNs", "Logistic Models", "EfficientNet", "Transformers (BERT/GPT)"]
}

df = pd.DataFrame(data)

print("\nActivation Function Comparison:\n")
print(df.to_string(index=False))

# --- Heatmap (Fully Fixed) ---
features_df = df.drop(columns=["Function", "Used in"])
# Set pandas option to opt into future behavior and silence warning
pd.set_option('future.no_silent_downcasting', True)

bool_df = features_df.replace({"Yes": 1, "No": 0})
bool_df = bool_df.astype("int64")  # Explicit cast avoids FutureWarning
annot_df = features_df.values      # Annotate with "Yes"/"No"

plt.figure(figsize=(10, 1.5))
sns.heatmap(bool_df, cmap="YlGnBu", annot=annot_df, fmt="s",
            cbar=False, xticklabels=features_df.columns, yticklabels=df["Function"])
plt.title("Feature Comparison (1 = Yes, 0 = No)")
plt.tight_layout()
plt.show()
