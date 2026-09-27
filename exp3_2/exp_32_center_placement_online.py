import numpy as np

from src.rbf import *
from src.data_setup import *


# ============================================================
# Configuration
# Best ONLINE model for noisy sine
# ============================================================

n_rbf = 15
sigma = 1.0
eta = 0.1

max_epochs = 1000
tolerance = 1e-5
patience = 5

n_random_runs = 20


# ============================================================
# 1. Evenly spaced RBF centers
# ============================================================

mus_manual = np.linspace(
    0,
    2 * np.pi,
    n_rbf
)

# Reproducible training order
np.random.seed(42)

weights_manual, epochs_manual = train_delta(
    x_train,
    sin_train_noisy,
    mus_manual,
    sigma,
    eta,
    max_epochs,
    tolerance=tolerance,
    patience=patience
)


# ------------------------------------------------------------
# Validation performance
# ------------------------------------------------------------

Phi_manual_val = design_matrix(
    x_val,
    mus_manual,
    sigma
)

predictions_manual_val = Phi_manual_val @ weights_manual

manual_val_error = residual_error(
    predictions_manual_val,
    sin_val_noisy
)


print("\n========== EVENLY SPACED CENTERS ==========")

print(f"Validation MAE: {manual_val_error:.5f}")
print(f"Epochs trained: {epochs_manual}")


# ============================================================
# 2. Randomly positioned RBF centers
# ============================================================

random_val_errors = []
random_epochs = []
random_centers = []

for run in range(n_random_runs):

    # Different reproducible center placement for every run
    rng = np.random.default_rng(123 + run)

    mus_random = rng.uniform(
        0,
        2 * np.pi,
        n_rbf
    )

    # Keep centers ordered for readability
    mus_random = np.sort(mus_random)

    # Reproducible training order
    np.random.seed(42 + run)

    weights_random, epochs_trained = train_delta(
        x_train,
        sin_train_noisy,
        mus_random,
        sigma,
        eta,
        max_epochs,
        tolerance=tolerance,
        patience=patience
    )

    # Validation predictions
    Phi_random_val = design_matrix(
        x_val,
        mus_random,
        sigma
    )

    predictions_random_val = Phi_random_val @ weights_random

    val_error = residual_error(
        predictions_random_val,
        sin_val_noisy
    )

    random_val_errors.append(val_error)
    random_epochs.append(epochs_trained)
    random_centers.append(mus_random)

    print(
        f"Run {run + 1:2d} | "
        f"Validation MAE: {val_error:.5f} | "
        f"Epochs: {epochs_trained}"
    )


# ============================================================
# 3. Random-center statistics
# ============================================================

mean_random_val = np.mean(random_val_errors)
std_random_val = np.std(random_val_errors)

mean_random_epochs = np.mean(random_epochs)
std_random_epochs = np.std(random_epochs)


print("\n========== RANDOM CENTER SUMMARY ==========")

print(
    f"Validation MAE: "
    f"{mean_random_val:.5f} ± {std_random_val:.5f}"
)

print(
    f"Epochs: "
    f"{mean_random_epochs:.1f} ± {std_random_epochs:.1f}"
)


# ============================================================
# 4. Compare center-placement strategies
# ============================================================

print("\n========== EVENLY SPACED VS RANDOM ==========")

print(
    f"Evenly spaced validation MAE: "
    f"{manual_val_error:.5f}"
)

print(
    f"Random validation MAE:        "
    f"{mean_random_val:.5f} ± {std_random_val:.5f}"
)


if manual_val_error < mean_random_val:
    print("\nEvenly spaced centers performed better on average.")
    selected_centers = mus_manual
    selected_strategy = "Evenly spaced"

else:
    print("\nRandom centers performed better on average.")

    # Select the random placement with the lowest validation error
    best_random_idx = np.argmin(random_val_errors)
    selected_centers = random_centers[best_random_idx]
    selected_strategy = "Random"


# ============================================================
# 5. Final test evaluation
# ============================================================

# Retrain selected center configuration
np.random.seed(42)

weights_final, epochs_final = train_delta(
    x_train,
    sin_train_noisy,
    selected_centers,
    sigma,
    eta,
    max_epochs,
    tolerance=tolerance,
    patience=patience
)

Phi_test = design_matrix(
    x_test,
    selected_centers,
    sigma
)

predictions_test = Phi_test @ weights_final


# Performance on noisy test data
noisy_test_error = residual_error(
    predictions_test,
    sin_test_noisy
)

# Performance on original clean function
clean_test_error = residual_error(
    predictions_test,
    sin_test
)


print("\n========== FINAL TEST PERFORMANCE ==========")

print(f"Selected strategy: {selected_strategy}")
print(f"RBF units:         {n_rbf}")
print(f"Sigma:             {sigma}")
print(f"Eta:               {eta}")
print(f"Epochs:            {epochs_final}")
print(f"Noisy test MAE:    {noisy_test_error:.5f}")
print(f"Clean test MAE:    {clean_test_error:.5f}")


