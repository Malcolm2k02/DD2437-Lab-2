from src.data_setup import *
from src.rbf import *

import numpy as np


# ============================================================
# Hyperparameters to investigate
# ============================================================

n_rbfs_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
sigmas = [0.1, 0.25, 0.5, 1.0, 1.5]


def find_best_batch(
    name,
    train_targets,
    val_targets,
    test_targets,
    clean_test_targets
):

    print(f"\n========== {name} ==========")

    best_val_error = float("inf")
    best_n = None
    best_sigma = None
    best_weights = None

    # ========================================================
    # GRID SEARCH
    # Select hyperparameters using VALIDATION data
    # ========================================================

    for sigma in sigmas:

        for n_rbfs in n_rbfs_list:

            # Uniformly spaced RBF centers
            mus = np.linspace(
                0,
                2 * np.pi,
                n_rbfs
            )

            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            Phi_train = design_matrix(
                x_train,
                mus,
                sigma
            )

            weights = train_least_squares(
                Phi_train,
                train_targets
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            Phi_val = design_matrix(
                x_val,
                mus,
                sigma
            )

            val_predictions = Phi_val @ weights

            val_error = residual_error(
                val_predictions,
                val_targets
            )

            print(
                f"RBFs={n_rbfs:3d}, "
                f"sigma={sigma:.2f}, "
                f"Validation MAE={val_error:.5f}"
            )

            # Select model using VALIDATION error
            if val_error < best_val_error:

                best_val_error = val_error
                best_n = n_rbfs
                best_sigma = sigma
                best_weights = weights.copy()


    # ========================================================
    # FINAL TEST
    # Test ONLY the selected configuration
    # ========================================================

    best_mus = np.linspace(
        0,
        2 * np.pi,
        best_n
    )

    Phi_test = design_matrix(
        x_test,
        best_mus,
        best_sigma
    )

    test_predictions = Phi_test @ best_weights

    # Performance against noisy test targets
    noisy_test_error = residual_error(
        test_predictions,
        test_targets
    )

    # Performance against original clean test targets
    clean_test_error = residual_error(
        test_predictions,
        clean_test_targets
    )


    # ========================================================
    # Results
    # ========================================================

    print("\nBEST CONFIGURATION")
    print(f"RBFs: {best_n}")
    print(f"Sigma: {best_sigma}")
    print(f"Validation MAE: {best_val_error:.5f}")

    print("\nFINAL TEST PERFORMANCE")
    print(f"Noisy test MAE: {noisy_test_error:.5f}")
    print(f"Clean test MAE: {clean_test_error:.5f}")

    return (
        best_n,
        best_sigma,
        best_val_error,
        noisy_test_error,
        clean_test_error
    )


# ============================================================
# Noisy sine
# ============================================================

(
    best_sin_n,
    best_sin_sigma,
    best_sin_val,
    best_sin_noisy_test,
    best_sin_clean_test
) = find_best_batch(

    name="NOISY SINE",

    train_targets=sin_train_noisy,
    val_targets=sin_val_noisy,

    test_targets=sin_test_noisy,
    clean_test_targets=sin_test
)


# ============================================================
# Noisy square
# ============================================================

(
    best_square_n,
    best_square_sigma,
    best_square_val,
    best_square_noisy_test,
    best_square_clean_test
) = find_best_batch(

    name="NOISY SQUARE",

    train_targets=square_train_noisy,
    val_targets=square_val_noisy,

    test_targets=square_test_noisy,
    clean_test_targets=square_test
)


# ============================================================
# Final summary
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print("\nSINE")
print(f"RBFs: {best_sin_n}")
print(f"Sigma: {best_sin_sigma}")
print(f"Validation MAE: {best_sin_val:.5f}")
print(f"Noisy test MAE: {best_sin_noisy_test:.5f}")
print(f"Clean test MAE: {best_sin_clean_test:.5f}")

print("\nSQUARE")
print(f"RBFs: {best_square_n}")
print(f"Sigma: {best_square_sigma}")
print(f"Validation MAE: {best_square_val:.5f}")
print(f"Noisy test MAE: {best_square_noisy_test:.5f}")
print(f"Clean test MAE: {best_square_clean_test:.5f}")


"""
SINE
RBFs: 10
Sigma: 1.0
Validation MAE: 0.17623
Noisy test MAE: 0.24998
Clean test MAE: 0.08693

SQUARE
RBFs: 20
Sigma: 0.5
Validation MAE: 0.36713
Noisy test MAE: 0.38191
Clean test MAE: 0.29037
"""

"""

========== NOISY SINE ==========
RBFs= 10, sigma=0.10, Validation MAE=0.80334
RBFs= 15, sigma=0.10, Validation MAE=0.77027
RBFs= 20, sigma=0.10, Validation MAE=0.46564
RBFs= 25, sigma=0.10, Validation MAE=0.40519
RBFs= 30, sigma=0.10, Validation MAE=0.44835
RBFs= 35, sigma=0.10, Validation MAE=0.50198
RBFs= 40, sigma=0.10, Validation MAE=0.66441
RBFs= 45, sigma=0.10, Validation MAE=6.84196
RBFs= 50, sigma=0.10, Validation MAE=11539.82628
RBFs=100, sigma=0.10, Validation MAE=0.84862
RBFs= 10, sigma=0.25, Validation MAE=0.30860
RBFs= 15, sigma=0.25, Validation MAE=0.23648
RBFs= 20, sigma=0.25, Validation MAE=0.26416
RBFs= 25, sigma=0.25, Validation MAE=0.30272
RBFs= 30, sigma=0.25, Validation MAE=0.40427
RBFs= 35, sigma=0.25, Validation MAE=0.42000
RBFs= 40, sigma=0.25, Validation MAE=0.65354
RBFs= 45, sigma=0.25, Validation MAE=3.54774
RBFs= 50, sigma=0.25, Validation MAE=16.76161
RBFs=100, sigma=0.25, Validation MAE=39.51615
RBFs= 10, sigma=0.50, Validation MAE=0.18666
RBFs= 15, sigma=0.50, Validation MAE=0.22624
RBFs= 20, sigma=0.50, Validation MAE=0.32262
RBFs= 25, sigma=0.50, Validation MAE=0.40361
RBFs= 30, sigma=0.50, Validation MAE=0.90266
RBFs= 35, sigma=0.50, Validation MAE=0.72214
RBFs= 40, sigma=0.50, Validation MAE=5.34517
RBFs= 45, sigma=0.50, Validation MAE=6.24631
RBFs= 50, sigma=0.50, Validation MAE=6.76901
RBFs=100, sigma=0.50, Validation MAE=7.92792
RBFs= 10, sigma=1.00, Validation MAE=0.17623
RBFs= 15, sigma=1.00, Validation MAE=0.26311
RBFs= 20, sigma=1.00, Validation MAE=0.71397
RBFs= 25, sigma=1.00, Validation MAE=0.32991
RBFs= 30, sigma=1.00, Validation MAE=0.32306
RBFs= 35, sigma=1.00, Validation MAE=0.31951
RBFs= 40, sigma=1.00, Validation MAE=0.31773
RBFs= 45, sigma=1.00, Validation MAE=0.31707
RBFs= 50, sigma=1.00, Validation MAE=0.31639
RBFs=100, sigma=1.00, Validation MAE=0.31545
RBFs= 10, sigma=1.50, Validation MAE=0.18771
RBFs= 15, sigma=1.50, Validation MAE=0.39777
RBFs= 20, sigma=1.50, Validation MAE=0.53211
RBFs= 25, sigma=1.50, Validation MAE=0.53243
RBFs= 30, sigma=1.50, Validation MAE=0.53272
RBFs= 35, sigma=1.50, Validation MAE=0.53292
RBFs= 40, sigma=1.50, Validation MAE=0.53295
RBFs= 45, sigma=1.50, Validation MAE=0.53300
RBFs= 50, sigma=1.50, Validation MAE=0.53312
RBFs=100, sigma=1.50, Validation MAE=0.53305

BEST CONFIGURATION
RBFs: 10
Sigma: 1.0
Validation MAE: 0.17623

FINAL TEST PERFORMANCE
Noisy test MAE: 0.24998
Clean test MAE: 0.08693

========== NOISY SQUARE ==========
RBFs= 10, sigma=0.10, Validation MAE=0.81303
RBFs= 15, sigma=0.10, Validation MAE=0.80891
RBFs= 20, sigma=0.10, Validation MAE=0.57390
RBFs= 25, sigma=0.10, Validation MAE=0.61759
RBFs= 30, sigma=0.10, Validation MAE=0.71499
RBFs= 35, sigma=0.10, Validation MAE=2.26074
RBFs= 40, sigma=0.10, Validation MAE=1.32733
RBFs= 45, sigma=0.10, Validation MAE=3.53251
RBFs= 50, sigma=0.10, Validation MAE=4756.94243
RBFs=100, sigma=0.10, Validation MAE=0.85334
RBFs= 10, sigma=0.25, Validation MAE=0.39008
RBFs= 15, sigma=0.25, Validation MAE=0.43981
RBFs= 20, sigma=0.25, Validation MAE=0.42606
RBFs= 25, sigma=0.25, Validation MAE=0.50253
RBFs= 30, sigma=0.25, Validation MAE=0.58834
RBFs= 35, sigma=0.25, Validation MAE=0.81682
RBFs= 40, sigma=0.25, Validation MAE=1.10082
RBFs= 45, sigma=0.25, Validation MAE=2.04508
RBFs= 50, sigma=0.25, Validation MAE=15.04887
RBFs=100, sigma=0.25, Validation MAE=37.15630
RBFs= 10, sigma=0.50, Validation MAE=0.39876
RBFs= 15, sigma=0.50, Validation MAE=0.43516
RBFs= 20, sigma=0.50, Validation MAE=0.36713
RBFs= 25, sigma=0.50, Validation MAE=0.65576
RBFs= 30, sigma=0.50, Validation MAE=1.40114
RBFs= 35, sigma=0.50, Validation MAE=3.56768
RBFs= 40, sigma=0.50, Validation MAE=1.76516
RBFs= 45, sigma=0.50, Validation MAE=0.95553
RBFs= 50, sigma=0.50, Validation MAE=0.48354
RBFs=100, sigma=0.50, Validation MAE=1.30162
RBFs= 10, sigma=1.00, Validation MAE=0.39843
RBFs= 15, sigma=1.00, Validation MAE=0.60666
RBFs= 20, sigma=1.00, Validation MAE=0.52163
RBFs= 25, sigma=1.00, Validation MAE=1.42731
RBFs= 30, sigma=1.00, Validation MAE=1.43880
RBFs= 35, sigma=1.00, Validation MAE=1.44411
RBFs= 40, sigma=1.00, Validation MAE=1.44737
RBFs= 45, sigma=1.00, Validation MAE=1.44899
RBFs= 50, sigma=1.00, Validation MAE=1.45012
RBFs=100, sigma=1.00, Validation MAE=1.45185
RBFs= 10, sigma=1.50, Validation MAE=0.40278
RBFs= 15, sigma=1.50, Validation MAE=0.65985
RBFs= 20, sigma=1.50, Validation MAE=0.50507
RBFs= 25, sigma=1.50, Validation MAE=0.50759
RBFs= 30, sigma=1.50, Validation MAE=0.50863
RBFs= 35, sigma=1.50, Validation MAE=0.50919
RBFs= 40, sigma=1.50, Validation MAE=0.50945
RBFs= 45, sigma=1.50, Validation MAE=0.50959
RBFs= 50, sigma=1.50, Validation MAE=0.50974
RBFs=100, sigma=1.50, Validation MAE=0.50983

BEST CONFIGURATION
RBFs: 20
Sigma: 0.5
Validation MAE: 0.36713

FINAL TEST PERFORMANCE
Noisy test MAE: 0.38191
Clean test MAE: 0.29037

========== FINAL SUMMARY ==========

SINE
RBFs: 10
Sigma: 1.0
Validation MAE: 0.17623
Noisy test MAE: 0.24998
Clean test MAE: 0.08693

SQUARE
RBFs: 20
Sigma: 0.5
Validation MAE: 0.36713
Noisy test MAE: 0.38191
Clean test MAE: 0.29037"""