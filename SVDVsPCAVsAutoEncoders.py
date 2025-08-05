import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load full MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize to [0, 1] and flatten
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# 2. PCA
pca = PCA(n_components=100)
pca.fit(x_train_flat)
x_pca_recon = pca.inverse_transform(pca.transform(x_test_flat))

# 3. SVD
def svd_reconstruct(X, k):
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
    X_recon = np.dot(U_k, np.dot(np.diag(S_k), VT_k)) + X_mean
    return X_recon

x_svd_recon = svd_reconstruct(x_test_flat, k=100)

# 4. Autoencoder
autoencoder = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(100, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train_flat, x_train_flat, epochs=10, batch_size=256, shuffle=True, verbose=1)

x_ae_recon = autoencoder.predict(x_test_flat)

# 5. Print MSE
print("Reconstruction MSE:")
print("PCA         :", mean_squared_error(x_test_flat, x_pca_recon))
print("SVD         :", mean_squared_error(x_test_flat, x_svd_recon))
print("Autoencoder :", mean_squared_error(x_test_flat, x_ae_recon))

# 6. Visualize reconstructions
def plot_digits(original, pca, svd, ae, n=5):
    plt.figure(figsize=(12, 5))
    for i in range(n):
        for j, img_set in enumerate([original, pca, svd, ae]):
            ax = plt.subplot(n, 4, i * 4 + j + 1)
            plt.imshow(img_set[i].reshape(28, 28), cmap='gray')
            plt.axis('off')
            if i == 0:
                ax.set_title(['Original', 'PCA', 'SVD', 'Autoencoder'][j])
    plt.tight_layout()
    plt.show()

plot_digits(x_test_flat, x_pca_recon, x_svd_recon, x_ae_recon)
