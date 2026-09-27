import numpy as np

from src.rbf import *
from src.data_setup import *


# ============================================================
# Configuration
# Best BATCH model for noisy sine
# ============================================================

n_rbf = 10
sigma = 1.0

n_random_runs = 20


# ============================================================
# 1. Evenly spaced RBF centers
# ============================================================

mus_manual = np.linspace(
    0,
    2 * np.pi,
    n_rbf
)

# Training design matrix
Phi_manual_train = design_matrix(
    x_train,
    mus_manual,
    sigma
)

# Batch least-squares training
weights_manual = train_least_squares(
    Phi_manual_train,
    sin_train_noisy
)


# Validation performance
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


# ============================================================
# 2. Randomly positioned RBF centers
# ============================================================

random_val_errors = []
random_centers = []

for run in range(n_random_runs):

    # Reproducible but different random centers for each run
    rng = np.random.default_rng(123 + run)

    mus_random = np.sort(
        rng.uniform(
            0,
            2 * np.pi,
            n_rbf
        )
    )

    # Train
    Phi_random_train = design_matrix(
        x_train,
        mus_random,
        sigma
    )

    weights_random = train_least_squares(
        Phi_random_train,
        sin_train_noisy
    )

    # Validate
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
    random_centers.append(mus_random)

    print(
        f"Run {run + 1:2d} | "
        f"Validation MAE: {val_error:.5f}"
    )


# ============================================================
# 3. Random-center statistics
# ============================================================

mean_random_val = np.mean(random_val_errors)
std_random_val = np.std(random_val_errors)


print("\n========== RANDOM CENTER SUMMARY ==========")

print(
    f"Validation MAE: "
    f"{mean_random_val:.5f} ± {std_random_val:.5f}"
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

    # Select random placement with lowest validation MAE
    best_random_idx = np.argmin(random_val_errors)

    selected_centers = random_centers[best_random_idx]
    selected_strategy = "Random"


# ============================================================
# 5. Final test evaluation
# ============================================================

# Retrain using selected center configuration
Phi_final_train = design_matrix(
    x_train,
    selected_centers,
    sigma
)

weights_final = train_least_squares(
    Phi_final_train,
    sin_train_noisy
)


# Test predictions
Phi_test = design_matrix(
    x_test,
    selected_centers,
    sigma
)

predictions_test = Phi_test @ weights_final


# Noisy test performance
noisy_test_error = residual_error(
    predictions_test,
    sin_test_noisy
)

# Original clean function
clean_test_error = residual_error(
    predictions_test,
    sin_test
)


print("\n========== FINAL TEST PERFORMANCE ==========")

print(f"Selected strategy: {selected_strategy}")
print(f"RBF units:         {n_rbf}")
print(f"Sigma:             {sigma}")
print(f"Noisy test MAE:    {noisy_test_error:.5f}")
print(f"Clean test MAE:    {clean_test_error:.5f}")