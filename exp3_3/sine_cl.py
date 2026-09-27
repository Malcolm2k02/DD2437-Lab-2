import numpy as np
import matplotlib.pyplot as plt

from src.data_setup import *
from src.rbf import residual_error, train_delta_variable_sigma, design_matrix_variable_sigma
from competitive_learning import competitive_learning, calculate_sigmas


# ============================================================
# Settings
# ============================================================

n_units = 10        # Replace with best architecture from 3.1
eta_cl = 0.1
eta_delta = 0.01
cl_epochs = 100
delta_epochs = 1000


# ============================================================
# 1. Competitive learning
# ============================================================

centers_cl, wins = competitive_learning(
    x_train,
    n_units=n_units,
    eta=eta_cl,
    epochs=cl_epochs,
    seed=42
)

centers_cl = centers_cl.flatten()

# Sort only for easier inspection/plotting
centers_cl = np.sort(centers_cl)


# ============================================================
# 2. Determine widths
# ============================================================

sigmas_cl = calculate_sigmas(centers_cl)


# ============================================================
# 3. Train output weights using delta learning
# ============================================================

weights_cl, epochs_used = train_delta_variable_sigma(
    x_train,
    sin_train,
    centers_cl,
    sigmas_cl,
    eta=eta_delta,
    max_epochs=delta_epochs
)


# ============================================================
# 4. Test predictions
# ============================================================

Phi_test = design_matrix_variable_sigma(
    x_test,
    centers_cl,
    sigmas_cl
)

predictions_cl = Phi_test @ weights_cl

test_mae = residual_error(
    predictions_cl,
    sin_test
)


# ============================================================
# Results
# ============================================================

print("\n========== CL RBF - CLEAN SINE ==========")

print("Centers:")
print(centers_cl)

print("\nSigmas:")
print(sigmas_cl)

print("\nWins:")
print(wins)

print(f"\nEpochs:   {epochs_used}")
print(f"Test MAE: {test_mae:.5f}")


# ============================================================
# Plot
# ============================================================

order = np.argsort(x_test)

plt.figure(figsize=(9, 5))

plt.plot(
    x_test[order],
    sin_test[order],
    label="Target",
    linewidth=2
)

plt.plot(
    x_test[order],
    predictions_cl[order],
    "--",
    label="CL-RBF prediction",
    linewidth=2
)

plt.scatter(
    centers_cl,
    np.zeros_like(centers_cl),
    marker="x",
    s=80,
    label="CL centers"
)

plt.xlabel("x")
plt.ylabel("sin(2x)")
plt.title("RBF centers obtained using competitive learning")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()