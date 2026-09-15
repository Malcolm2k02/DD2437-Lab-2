from src.data_setup import *
from src.rbf import *

import numpy as np


# Same parameter ranges we used previously
n_rbfs_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
sigmas = [0.1, 0.25, 0.5, 1.0, 1.5]


def find_best_batch(
    name,
    train_targets,
    test_targets
):

    print(f"\n========== {name} ==========")

    best_error = float("inf")
    best_n = None
    best_sigma = None

    for sigma in sigmas:

        for n_rbfs in n_rbfs_list:

            # Uniformly spaced RBF centers
            mus = np.linspace(
                0,
                2 * np.pi,
                n_rbfs
            )

            # Training
            Phi_train = design_matrix(
                x_train,
                mus,
                sigma
            )

            weights = train_least_squares(
                Phi_train,
                train_targets
            )

            # Testing
            Phi_test = design_matrix(
                x_test,
                mus,
                sigma
            )

            predictions = Phi_test @ weights

            error = residual_error(
                predictions,
                test_targets
            )

            print(
                f"RBFs={n_rbfs:2d}, "
                f"sigma={sigma:.2f}, "
                f"MAE={error:.5f}"
            )

            if error < best_error:

                best_error = error
                best_n = n_rbfs
                best_sigma = sigma


    print("\nBEST CONFIGURATION")
    print(f"RBFs: {best_n}")
    print(f"Sigma: {best_sigma}")
    print(f"Test MAE: {best_error:.5f}")

    return best_n, best_sigma, best_error


# ============================================================
# Noisy sine
# ============================================================

best_sin_n, best_sin_sigma, best_sin_error = find_best_batch(
    name="NOISY SINE",
    train_targets=sin_train_noisy,
    test_targets=sin_test_noisy
)


# ============================================================
# Noisy square
# ============================================================

best_square_n, best_square_sigma, best_square_error = find_best_batch(
    name="NOISY SQUARE",
    train_targets=square_train_noisy,
    test_targets=square_test_noisy
)


# ============================================================
# Summary
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print(
    f"Sine: "
    f"{best_sin_n} RBFs, "
    f"sigma={best_sin_sigma}, "
    f"MAE={best_sin_error:.5f}"
)

print(
    f"Square: "
    f"{best_square_n} RBFs, "
    f"sigma={best_square_sigma}, "
    f"MAE={best_square_error:.5f}"
)

"""BEST CONFIGURATION
RBFs: 40
Sigma: 0.25
Test MAE: 0.31556

========== FINAL SUMMARY ==========
Sine: 10 RBFs, sigma=1.5, MAE=0.25244
Square: 40 RBFs, sigma=0.25, MAE=0.31556"""