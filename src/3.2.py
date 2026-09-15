import numpy as np
import matplotlib.pyplot as plt
from rbf import *

np.random.seed(42)  # For reproducibility

x_train = np.arange(0, 2 * np.pi, 0.1)

x_test = np.arange(0.05, 2 * np.pi, 0.1)

sin_train = np.sin(2 * x_train)
sin_test = np.sin(2 * x_test)

square_train = np.where(np.sin(2 * x_train) >= 0, 1, -1)
square_test = np.where(np.sin(2 * x_test) >= 0, 1, -1)

noise_var = 0.1
noise_std = np.sqrt(noise_var)

sin_train_noisy = sin_train + np.random.normal(0, noise_std, size=sin_train.shape)
sin_test_noisy = sin_test + np.random.normal(0, noise_std, size=sin_test.shape)

square_train_noisy = square_train + np.random.normal(0, noise_std, size=square_train.shape)
square_test_noisy = square_test + np.random.normal(0, noise_std, size=square_test.shape)


n_rbfs = [10, 15, 20, 25, 30, 35, 40, 45, 50]
sigmas = [0.1, 0.25, 0.5, 1.0, 1.5]
etas = [0.001, 0.01, 0.05, 0.1, 0.5]
epochs = 300
"""
Best configuration:
RBFs: 50
Sigma: 1.0
Eta: 0.05
Error: 0.242445
"""

results = []
"""
for sigma in sigmas:
    for eta in etas:
        for n_rbf in n_rbfs:

            mus = np.linspace(0, 2 * np.pi, n_rbf)

            weights_delta, epochs_trained = train_delta(
                x_train,
                sin_train_noisy,
                mus,
                sigma,
                eta,
                epochs,
                tolerance=1e-5,
                patience=5
            )

            Phi_test = design_matrix(
                x_test,
                mus,
                sigma
            )

            predictions_delta = Phi_test @ weights_delta

            error_delta = residual_error(
                predictions_delta,
                sin_test_noisy
            )

            results.append({
                "n_rbf": n_rbf,
                "sigma": sigma,
                "eta": eta,
                "error": error_delta
            })

            print(
                f"RBFs={n_rbf:2d} | "
                f"sigma={sigma:.2f} | "
                f"eta={eta:.3f} | "
                f"epochs={epochs_trained:3d} | "
                f"error={error_delta:.5f}"
            )

best = min(results, key=lambda r: r["error"])

print("\nBest configuration:")
print(f"RBFs: {best['n_rbf']}")
print(f"Sigma: {best['sigma']}")
print(f"Eta: {best['eta']}")
print(f"Error: {best['error']:.6f}")
"""

rbf = 50
sigma = 1.0
eta = 0.05
mus = np.linspace(0, 2 * np.pi, rbf)
mus_random = np.random.uniform(0, 2 * np.pi, rbf)

weights_delta, epochs_trained = train_delta(
    x_train,
    sin_train_noisy,
    mus,
    sigma,
    eta,
    epochs,
    tolerance=1e-3,
    patience=5
    )

weights_random, epochs_random = train_delta(
    x_train,
    sin_train_noisy,
    mus_random,
    sigma,
    eta,
    epochs,
    tolerance=1e-3,
    patience=5
)

Phi_test = design_matrix(
    x_test,
    mus,
    sigma
    )

Phi_random = design_matrix(
    x_test,
    mus_random,
    sigma
)

predictions_delta = Phi_test @ weights_delta
predictions_random = Phi_random @ weights_random

noisy_error = residual_error(
    predictions_delta,
    sin_test_noisy
)

clean_error = residual_error(
    predictions_delta,
    sin_test
)

error_random = residual_error(
    predictions_random,
    sin_test_noisy
)

error_random_clean = residual_error(
    predictions_random,
    sin_test
)

print("Noisy test error:", noisy_error, "Epochs trained:", epochs_trained)
print("Clean test error:", clean_error, "Epochs trained:", epochs_trained)
print("Random test error:", error_random, "Epochs trained:", epochs_random)
print("Random clean test error:", error_random_clean, "Epochs trained:", epochs_random)