import numpy as np

from src.rbf import *
from src.data_setup import *


# ============================================================
# Configuration
# ============================================================

n_rbf = 50
sigma = 1.0
eta = 0.05

max_epochs = 3000
tolerance = 1e-3
patience = 5

n_random_runs = 10


# ============================================================
# 1. Manual / evenly spaced RBF centers
# ============================================================

mus_manual = np.linspace(
    0,
    2 * np.pi,
    n_rbf
)

# Reset seed for reproducible training order
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

Phi_manual = design_matrix(
    x_test,
    mus_manual,
    sigma
)

predictions_manual = Phi_manual @ weights_manual


manual_noisy_error = residual_error(
    predictions_manual,
    sin_test_noisy
)

manual_clean_error = residual_error(
    predictions_manual,
    sin_test
)


print("\n========== MANUAL CENTERS ==========")

print(f"Noisy test error: {manual_noisy_error:.5f}")
print(f"Clean test error: {manual_clean_error:.5f}")
print(f"Epochs trained: {epochs_manual}")


# ============================================================
# 2. Randomly positioned RBF centers
# ============================================================

random_noisy_errors = []
random_clean_errors = []
random_epochs = []

# Different seed from manual experiment
np.random.seed(123)

for run in range(n_random_runs):

    # Random center positions between 0 and 2*pi
    mus_random = np.random.uniform(
        0,
        2 * np.pi,
        n_rbf
    )

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

    Phi_random = design_matrix(
        x_test,
        mus_random,
        sigma
    )

    predictions_random = Phi_random @ weights_random


    noisy_error = residual_error(
        predictions_random,
        sin_test_noisy
    )

    clean_error = residual_error(
        predictions_random,
        sin_test
    )


    random_noisy_errors.append(noisy_error)
    random_clean_errors.append(clean_error)
    random_epochs.append(epochs_trained)


    print(
        f"Run {run + 1:2d} | "
        f"Noisy error: {noisy_error:.5f} | "
        f"Clean error: {clean_error:.5f} | "
        f"Epochs: {epochs_trained}"
    )


# ============================================================
# 3. Random-center statistics
# ============================================================

mean_random_noisy = np.mean(random_noisy_errors)
std_random_noisy = np.std(random_noisy_errors)

mean_random_clean = np.mean(random_clean_errors)
std_random_clean = np.std(random_clean_errors)

mean_random_epochs = np.mean(random_epochs)


print("\n========== RANDOM CENTER SUMMARY ==========")

print(
    f"Noisy error: "
    f"{mean_random_noisy:.5f} ± {std_random_noisy:.5f}"
)

print(
    f"Clean error: "
    f"{mean_random_clean:.5f} ± {std_random_clean:.5f}"
)

print(
    f"Average epochs: "
    f"{mean_random_epochs:.1f}"
)


# ============================================================
# 4. Final comparison
# ============================================================

print("\n========== MANUAL VS RANDOM ==========")

print(
    f"Manual noisy error:       "
    f"{manual_noisy_error:.5f}"
)

print(
    f"Random mean noisy error:  "
    f"{mean_random_noisy:.5f}"
)

print()

print(
    f"Manual clean error:       "
    f"{manual_clean_error:.5f}"
)

print(
    f"Random mean clean error:  "
    f"{mean_random_clean:.5f}"
)


# Determine which performed better on the clean function

if manual_clean_error < mean_random_clean:
    print("\nManual centers performed better on average.")
else:
    print("\nRandom centers performed better on average.")


"""========== MANUAL CENTERS ==========
Noisy test error: 0.27470
Clean test error: 0.12784
Epochs trained: 3000
"""

"""========== RANDOM CENTER SUMMARY ==========
Run  1 | Noisy error: 0.26297 | Clean error: 0.11265 | Epochs: 3000
Run  2 | Noisy error: 0.24166 | Clean error: 0.09519 | Epochs: 3000
Run  3 | Noisy error: 0.32070 | Clean error: 0.20495 | Epochs: 3000
Run  4 | Noisy error: 0.25490 | Clean error: 0.13035 | Epochs: 3000
Run  5 | Noisy error: 0.31912 | Clean error: 0.22014 | Epochs: 3000
Run  6 | Noisy error: 0.26478 | Clean error: 0.10637 | Epochs: 3000
Run  7 | Noisy error: 0.26634 | Clean error: 0.13802 | Epochs: 3000
Run  8 | Noisy error: 0.31819 | Clean error: 0.21680 | Epochs: 3000
Run  9 | Noisy error: 0.41117 | Clean error: 0.36123 | Epochs: 3000
Run 10 | Noisy error: 0.45783 | Clean error: 0.38022 | Epochs: 3000

========== RANDOM CENTER SUMMARY ==========
Noisy error: 0.31177 ± 0.06796
Clean error: 0.19659 ± 0.09754
Average epochs: 3000.0

========== MANUAL VS RANDOM ==========
Manual noisy error:       0.27470
Random mean noisy error:  0.31177

Manual clean error:       0.12784
Random mean clean error:  0.19659

Manual centers performed better on average."""

