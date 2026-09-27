from src.data_setup import *
from src.rbf import *

import numpy as np


# ============================================================
# Hyperparameters
# ============================================================

n_rbfs_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
sigmas = [0.1, 0.25, 0.5, 1.0, 1.5]
etas = [0.001, 0.01, 0.05, 0.1, 0.5]


def find_best_online(
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
    best_eta = None
    best_weights = None
    best_epochs = None


    # ========================================================
    # GRID SEARCH
    # Hyperparameters selected using VALIDATION data
    # ========================================================

    for sigma in sigmas:

        for n_rbfs in n_rbfs_list:

            # Same uniform RBF positioning as batch experiment
            mus = np.linspace(
                0,
                2 * np.pi,
                n_rbfs
            )

            for eta in etas:

                # Makes shuffling reproducible
                np.random.seed(42)

                # --------------------------------------------
                # Online delta-rule training
                # --------------------------------------------

                weights, epochs_used = train_delta(
                    X=x_train,
                    targets=train_targets,
                    mus=mus,
                    sigma=sigma,
                    eta=eta,
                    max_epochs=1000,
                    tolerance=1e-5,
                    patience=5
                )

                # --------------------------------------------
                # Validation
                # --------------------------------------------

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
                    f"eta={eta:.3f}, "
                    f"epochs={epochs_used:4d}, "
                    f"Val MAE={val_error:.5f}"
                )

                # Select according to validation error
                if val_error < best_val_error:

                    best_val_error = val_error

                    best_n = n_rbfs
                    best_sigma = sigma
                    best_eta = eta

                    best_weights = weights.copy()
                    best_epochs = epochs_used


    # ========================================================
    # FINAL TEST
    # Test set has not been used for model selection
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


    # Test against noisy targets
    noisy_test_error = residual_error(
        test_predictions,
        test_targets
    )

    # Test against original clean targets
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
    print(f"Eta: {best_eta}")
    print(f"Epochs: {best_epochs}")
    print(f"Validation MAE: {best_val_error:.5f}")

    print("\nFINAL TEST PERFORMANCE")
    print(f"Noisy test MAE: {noisy_test_error:.5f}")
    print(f"Clean test MAE: {clean_test_error:.5f}")

    return (
        best_n,
        best_sigma,
        best_eta,
        best_epochs,
        best_val_error,
        noisy_test_error,
        clean_test_error
    )


# ============================================================
# NOISY SINE
# ============================================================

(
    best_sin_n,
    best_sin_sigma,
    best_sin_eta,
    best_sin_epochs,
    best_sin_val,
    best_sin_noisy_test,
    best_sin_clean_test
) = find_best_online(

    name="ONLINE - NOISY SINE",

    train_targets=sin_train_noisy,
    val_targets=sin_val_noisy,

    test_targets=sin_test_noisy,
    clean_test_targets=sin_test
)


# ============================================================
# NOISY SQUARE
# ============================================================

(
    best_square_n,
    best_square_sigma,
    best_square_eta,
    best_square_epochs,
    best_square_val,
    best_square_noisy_test,
    best_square_clean_test
) = find_best_online(

    name="ONLINE - NOISY SQUARE",

    train_targets=square_train_noisy,
    val_targets=square_val_noisy,

    test_targets=square_test_noisy,
    clean_test_targets=square_test
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== FINAL ONLINE SUMMARY ==========")

print("\nSINE")
print(f"RBFs: {best_sin_n}")
print(f"Sigma: {best_sin_sigma}")
print(f"Eta: {best_sin_eta}")
print(f"Epochs: {best_sin_epochs}")
print(f"Validation MAE: {best_sin_val:.5f}")
print(f"Noisy test MAE: {best_sin_noisy_test:.5f}")
print(f"Clean test MAE: {best_sin_clean_test:.5f}")

print("\nSQUARE")
print(f"RBFs: {best_square_n}")
print(f"Sigma: {best_square_sigma}")
print(f"Eta: {best_square_eta}")
print(f"Epochs: {best_square_epochs}")
print(f"Validation MAE: {best_square_val:.5f}")
print(f"Noisy test MAE: {best_square_noisy_test:.5f}")
print(f"Clean test MAE: {best_square_clean_test:.5f}")

"""
========== ONLINE - NOISY SINE ==========
RBFs= 10, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.53667
RBFs= 10, sigma=0.10, eta=0.010, epochs= 454, Val MAE=0.66392
RBFs= 10, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.77684
RBFs= 10, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.79821
RBFs= 10, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.79970
RBFs= 15, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.44129
RBFs= 15, sigma=0.10, eta=0.010, epochs= 124, Val MAE=0.42024
RBFs= 15, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.74730
RBFs= 15, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.76668
RBFs= 15, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.77174
RBFs= 20, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.25555
RBFs= 20, sigma=0.10, eta=0.010, epochs= 404, Val MAE=0.34851
RBFs= 20, sigma=0.10, eta=0.050, epochs= 179, Val MAE=0.38952
RBFs= 20, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.46117
RBFs= 20, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.46173
RBFs= 25, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.23906
RBFs= 25, sigma=0.10, eta=0.010, epochs= 247, Val MAE=0.23998
RBFs= 25, sigma=0.10, eta=0.050, epochs= 524, Val MAE=0.31761
RBFs= 25, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.38682
RBFs= 25, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.40515
RBFs= 30, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.21436
RBFs= 30, sigma=0.10, eta=0.010, epochs= 315, Val MAE=0.22384
RBFs= 30, sigma=0.10, eta=0.050, epochs= 113, Val MAE=0.23696
RBFs= 30, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.37285
RBFs= 30, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.45653
RBFs= 35, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.23139
RBFs= 35, sigma=0.10, eta=0.010, epochs= 422, Val MAE=0.27616
RBFs= 35, sigma=0.10, eta=0.050, epochs= 205, Val MAE=0.29957
RBFs= 35, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.36363
RBFs= 35, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.48957
RBFs= 40, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.21010
RBFs= 40, sigma=0.10, eta=0.010, epochs= 573, Val MAE=0.27603
RBFs= 40, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.38640
RBFs= 40, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.48837
RBFs= 40, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.62147
RBFs= 45, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.20462
RBFs= 45, sigma=0.10, eta=0.010, epochs= 974, Val MAE=0.30746
RBFs= 45, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.44430
RBFs= 45, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.61726
RBFs= 45, sigma=0.10, eta=0.500, epochs=1000, Val MAE=1.15413
RBFs= 50, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.19811
RBFs= 50, sigma=0.10, eta=0.010, epochs=1000, Val MAE=0.31325
RBFs= 50, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.38448
RBFs= 50, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.49346
RBFs= 50, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.96232
RBFs=100, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.24911
RBFs=100, sigma=0.10, eta=0.010, epochs=1000, Val MAE=0.32814
RBFs=100, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.48353
RBFs=100, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.60358
RBFs=100, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.84334
RBFs= 10, sigma=0.25, eta=0.001, epochs= 772, Val MAE=0.27785
RBFs= 10, sigma=0.25, eta=0.010, epochs= 172, Val MAE=0.30287
RBFs= 10, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.30805
RBFs= 10, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.30814
RBFs= 10, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.30365
RBFs= 15, sigma=0.25, eta=0.001, epochs= 898, Val MAE=0.20297
RBFs= 15, sigma=0.25, eta=0.010, epochs= 344, Val MAE=0.22248
RBFs= 15, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.23665
RBFs= 15, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.23686
RBFs= 15, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.30699
RBFs= 20, sigma=0.25, eta=0.001, epochs= 986, Val MAE=0.20690
RBFs= 20, sigma=0.25, eta=0.010, epochs= 645, Val MAE=0.22942
RBFs= 20, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.26155
RBFs= 20, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.26841
RBFs= 20, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.37232
RBFs= 25, sigma=0.25, eta=0.001, epochs= 999, Val MAE=0.20600
RBFs= 25, sigma=0.25, eta=0.010, epochs= 436, Val MAE=0.21027
RBFs= 25, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.23450
RBFs= 25, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.25834
RBFs= 25, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.39191
RBFs= 30, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.20660
RBFs= 30, sigma=0.25, eta=0.010, epochs= 986, Val MAE=0.20788
RBFs= 30, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.23281
RBFs= 30, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.25370
RBFs= 30, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.39298
RBFs= 35, sigma=0.25, eta=0.001, epochs= 967, Val MAE=0.20695
RBFs= 35, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.20876
RBFs= 35, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.23409
RBFs= 35, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.25822
RBFs= 35, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.42494
RBFs= 40, sigma=0.25, eta=0.001, epochs= 843, Val MAE=0.20702
RBFs= 40, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.20937
RBFs= 40, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.23548
RBFs= 40, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.26280
RBFs= 40, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.45538
RBFs= 45, sigma=0.25, eta=0.001, epochs= 761, Val MAE=0.20710
RBFs= 45, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.20999
RBFs= 45, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.23755
RBFs= 45, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.26737
RBFs= 45, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.47881
RBFs= 50, sigma=0.25, eta=0.001, epochs= 745, Val MAE=0.20723
RBFs= 50, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.21062
RBFs= 50, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.24040
RBFs= 50, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.27183
RBFs= 50, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.51042
RBFs=100, sigma=0.25, eta=0.001, epochs= 560, Val MAE=0.20866
RBFs=100, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.21685
RBFs=100, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.26720
RBFs=100, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.33379
C:\Users\malco\OneDrive\Dokument\GitHub\DD2437-Lab-2\src\rbf.py:77: RuntimeWarning: overflow encountered in matmul
  predictions = Phi @ weights
C:\Users\malco\anaconda3\envs\DD2437\Lib\site-packages\numpy\_core\_methods.py:132: RuntimeWarning: overflow encountered in reduce
  ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
C:\Users\malco\OneDrive\Dokument\GitHub\DD2437-Lab-2\src\rbf.py:69: RuntimeWarning: overflow encountered in matmul
  prediction = phi @ weights
C:\Users\malco\OneDrive\Dokument\GitHub\DD2437-Lab-2\src\rbf.py:73: RuntimeWarning: invalid value encountered in add
  weights += eta * error * phi
RBFs=100, sigma=0.25, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 10, sigma=0.50, eta=0.001, epochs= 905, Val MAE=0.20204
RBFs= 10, sigma=0.50, eta=0.010, epochs= 253, Val MAE=0.19387
RBFs= 10, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.18628
RBFs= 10, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.18916
RBFs= 10, sigma=0.50, eta=0.500, epochs=1000, Val MAE=0.31613
RBFs= 15, sigma=0.50, eta=0.001, epochs= 669, Val MAE=0.19943
RBFs= 15, sigma=0.50, eta=0.010, epochs= 653, Val MAE=0.18512
RBFs= 15, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.18729
RBFs= 15, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.19827
RBFs= 15, sigma=0.50, eta=0.500, epochs=1000, Val MAE=0.42127
RBFs= 20, sigma=0.50, eta=0.001, epochs= 520, Val MAE=0.19927
RBFs= 20, sigma=0.50, eta=0.010, epochs= 641, Val MAE=0.18492
RBFs= 20, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.18977
RBFs= 20, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.21818
RBFs= 20, sigma=0.50, eta=0.500, epochs=1000, Val MAE=0.52773
RBFs= 25, sigma=0.50, eta=0.001, epochs= 425, Val MAE=0.19922
RBFs= 25, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.18597
RBFs= 25, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.19431
RBFs= 25, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.23839
RBFs= 25, sigma=0.50, eta=0.500, epochs=1000, Val MAE=0.66528
RBFs= 30, sigma=0.50, eta=0.001, epochs= 359, Val MAE=0.19917
RBFs= 30, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.18659
RBFs= 30, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.20168
RBFs= 30, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.25843
RBFs= 30, sigma=0.50, eta=0.500, epochs=1000, Val MAE=67865258088318.92188
RBFs= 35, sigma=0.50, eta=0.001, epochs= 325, Val MAE=0.19898
RBFs= 35, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.18713
RBFs= 35, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.21102
RBFs= 35, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.27796
RBFs= 35, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 40, sigma=0.50, eta=0.001, epochs= 332, Val MAE=0.19781
RBFs= 40, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.18761
RBFs= 40, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.22069
RBFs= 40, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.29689
RBFs= 40, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 45, sigma=0.50, eta=0.001, epochs= 298, Val MAE=0.19771
RBFs= 45, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.18804
RBFs= 45, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.23055
RBFs= 45, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.31525
RBFs= 45, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 50, sigma=0.50, eta=0.001, epochs= 271, Val MAE=0.19777
RBFs= 50, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.18842
RBFs= 50, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.24049
RBFs= 50, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.33545
RBFs= 50, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs=100, sigma=0.50, eta=0.001, epochs= 253, Val MAE=0.19165
RBFs=100, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.19215
RBFs=100, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.33753
RBFs=100, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.54943
RBFs=100, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 10, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.36100
RBFs= 10, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.21119
RBFs= 10, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.18971
RBFs= 10, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.17971
RBFs= 10, sigma=1.00, eta=0.500, epochs=1000, Val MAE=0.31237
RBFs= 15, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.32998
RBFs= 15, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.20771
RBFs= 15, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.18381
RBFs= 15, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.17744
RBFs= 15, sigma=1.00, eta=0.500, epochs=1000, Val MAE=0.91101
RBFs= 20, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.31022
RBFs= 20, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.20493
RBFs= 20, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.18099
RBFs= 20, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.18202
RBFs= 20, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 25, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.29751
RBFs= 25, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.20301
RBFs= 25, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.17956
RBFs= 25, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.18999
RBFs= 25, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 30, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.28610
RBFs= 30, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.20103
RBFs= 30, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.17801
RBFs= 30, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.20044
RBFs= 30, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 35, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.27575
RBFs= 35, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.19909
RBFs= 35, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.17935
RBFs= 35, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.21747
RBFs= 35, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 40, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.26674
RBFs= 40, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.19726
RBFs= 40, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.18194
RBFs= 40, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.24159
RBFs= 40, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 45, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.26072
RBFs= 45, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.19554
RBFs= 45, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.18550
RBFs= 45, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.28618
RBFs= 45, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 50, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.25520
RBFs= 50, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.19391
RBFs= 50, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.19019
RBFs= 50, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.37909
RBFs= 50, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs=100, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.22039
RBFs=100, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.18235
RBFs=100, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.38907
RBFs=100, sigma=1.00, eta=0.100, epochs=1000, Val MAE=nan
RBFs=100, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 10, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.57850
RBFs= 10, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.50888
RBFs= 10, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.43124
RBFs= 10, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.36019
RBFs= 10, sigma=1.50, eta=0.500, epochs=1000, Val MAE=0.62215
RBFs= 15, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.55002
RBFs= 15, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.49551
RBFs= 15, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.40315
RBFs= 15, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.36182
RBFs= 15, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 20, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.53223
RBFs= 20, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.48533
RBFs= 20, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.38557
RBFs= 20, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.35472
RBFs= 20, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 25, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.52444
RBFs= 25, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.47711
RBFs= 25, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.38409
RBFs= 25, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.36169
RBFs= 25, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 30, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.51952
RBFs= 30, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.47027
RBFs= 30, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.38143
RBFs= 30, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.41342
RBFs= 30, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 35, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.51606
RBFs= 35, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.46407
RBFs= 35, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.37652
RBFs= 35, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.52329
RBFs= 35, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 40, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.51352
RBFs= 40, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.45834
RBFs= 40, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.37104
RBFs= 40, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.70831
RBFs= 40, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 45, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.51156
RBFs= 45, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.45296
RBFs= 45, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.36830
RBFs= 45, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.85534
RBFs= 45, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 50, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.50998
RBFs= 50, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.44785
RBFs= 50, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.37150
RBFs= 50, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.61951
RBFs= 50, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs=100, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.50009
RBFs=100, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.40840
RBFs=100, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.62779
RBFs=100, sigma=1.50, eta=0.100, epochs=1000, Val MAE=nan
RBFs=100, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan

BEST CONFIGURATION
RBFs: 15
Sigma: 1.0
Eta: 0.1
Epochs: 1000
Validation MAE: 0.17744

FINAL TEST PERFORMANCE
Noisy test MAE: 0.25127
Clean test MAE: 0.06503

========== ONLINE - NOISY SQUARE ==========
RBFs= 10, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.66448
RBFs= 10, sigma=0.10, eta=0.010, epochs= 420, Val MAE=0.76270
RBFs= 10, sigma=0.10, eta=0.050, epochs= 863, Val MAE=0.80423
RBFs= 10, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.81120
RBFs= 10, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.81227
RBFs= 15, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.67530
RBFs= 15, sigma=0.10, eta=0.010, epochs= 579, Val MAE=0.66446
RBFs= 15, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.79904
RBFs= 15, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.80788
RBFs= 15, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.80784
RBFs= 20, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.53770
RBFs= 20, sigma=0.10, eta=0.010, epochs= 408, Val MAE=0.43492
RBFs= 20, sigma=0.10, eta=0.050, epochs= 236, Val MAE=0.46105
RBFs= 20, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.56824
RBFs= 20, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.57997
RBFs= 25, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.47665
RBFs= 25, sigma=0.10, eta=0.010, epochs= 264, Val MAE=0.39504
RBFs= 25, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.53433
RBFs= 25, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.58950
RBFs= 25, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.60583
RBFs= 30, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.43338
RBFs= 30, sigma=0.10, eta=0.010, epochs= 211, Val MAE=0.39498
RBFs= 30, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.53267
RBFs= 30, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.60981
RBFs= 30, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.69819
RBFs= 35, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.43153
RBFs= 35, sigma=0.10, eta=0.010, epochs= 630, Val MAE=0.39447
RBFs= 35, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.67532
RBFs= 35, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.96441
RBFs= 35, sigma=0.10, eta=0.500, epochs=1000, Val MAE=2.07000
RBFs= 40, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.41914
RBFs= 40, sigma=0.10, eta=0.010, epochs=1000, Val MAE=0.45448
RBFs= 40, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.72257
RBFs= 40, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.90801
RBFs= 40, sigma=0.10, eta=0.500, epochs=1000, Val MAE=1.23917
RBFs= 45, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.41334
RBFs= 45, sigma=0.10, eta=0.010, epochs=1000, Val MAE=0.46855
RBFs= 45, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.73955
RBFs= 45, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.86811
RBFs= 45, sigma=0.10, eta=0.500, epochs=1000, Val MAE=1.08337
RBFs= 50, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.40983
RBFs= 50, sigma=0.10, eta=0.010, epochs=1000, Val MAE=0.45460
RBFs= 50, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.77049
RBFs= 50, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.88786
RBFs= 50, sigma=0.10, eta=0.500, epochs=1000, Val MAE=1.32596
RBFs=100, sigma=0.10, eta=0.001, epochs=1000, Val MAE=0.40053
RBFs=100, sigma=0.10, eta=0.010, epochs=1000, Val MAE=0.54460
RBFs=100, sigma=0.10, eta=0.050, epochs=1000, Val MAE=0.77106
RBFs=100, sigma=0.10, eta=0.100, epochs=1000, Val MAE=0.80448
RBFs=100, sigma=0.10, eta=0.500, epochs=1000, Val MAE=0.85164
RBFs= 10, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.39442
RBFs= 10, sigma=0.25, eta=0.010, epochs= 123, Val MAE=0.39462
RBFs= 10, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.39055
RBFs= 10, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.39134
RBFs= 10, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.40428
RBFs= 15, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.34871
RBFs= 15, sigma=0.25, eta=0.010, epochs= 554, Val MAE=0.42460
RBFs= 15, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.44063
RBFs= 15, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.44166
RBFs= 15, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.45810
RBFs= 20, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.32391
RBFs= 20, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.40462
RBFs= 20, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.42775
RBFs= 20, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.42584
RBFs= 20, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.41889
RBFs= 25, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.31885
RBFs= 25, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.40362
RBFs= 25, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.42774
RBFs= 25, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.44132
RBFs= 25, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.49047
RBFs= 30, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.32109
RBFs= 30, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.40325
RBFs= 30, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.42395
RBFs= 30, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.43531
RBFs= 30, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.50853
RBFs= 35, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.32937
RBFs= 35, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.40578
RBFs= 35, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.42482
RBFs= 35, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.43613
RBFs= 35, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.51652
RBFs= 40, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.33634
RBFs= 40, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.40787
RBFs= 40, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.42580
RBFs= 40, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.43825
RBFs= 40, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.57539
RBFs= 45, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.34226
RBFs= 45, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.40954
RBFs= 45, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.42675
RBFs= 45, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.43998
RBFs= 45, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.65455
RBFs= 50, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.34731
RBFs= 50, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.41088
RBFs= 50, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.42765
RBFs= 50, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.44141
RBFs= 50, sigma=0.25, eta=0.500, epochs=1000, Val MAE=0.79734
RBFs=100, sigma=0.25, eta=0.001, epochs=1000, Val MAE=0.37360
RBFs=100, sigma=0.25, eta=0.010, epochs=1000, Val MAE=0.41694
RBFs=100, sigma=0.25, eta=0.050, epochs=1000, Val MAE=0.43668
RBFs=100, sigma=0.25, eta=0.100, epochs=1000, Val MAE=0.45401
RBFs=100, sigma=0.25, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 10, sigma=0.50, eta=0.001, epochs= 471, Val MAE=0.36553
RBFs= 10, sigma=0.50, eta=0.010, epochs= 407, Val MAE=0.39461
RBFs= 10, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.40062
RBFs= 10, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.40733
RBFs= 10, sigma=0.50, eta=0.500, epochs=1000, Val MAE=0.52507
RBFs= 15, sigma=0.50, eta=0.001, epochs= 340, Val MAE=0.36194
RBFs= 15, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.37658
RBFs= 15, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.35459
RBFs= 15, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.36346
RBFs= 15, sigma=0.50, eta=0.500, epochs=1000, Val MAE=0.63720
RBFs= 20, sigma=0.50, eta=0.001, epochs= 253, Val MAE=0.36166
RBFs= 20, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.37254
RBFs= 20, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.34980
RBFs= 20, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.37241
RBFs= 20, sigma=0.50, eta=0.500, epochs=1000, Val MAE=0.68267
RBFs= 25, sigma=0.50, eta=0.001, epochs= 223, Val MAE=0.36171
RBFs= 25, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.36902
RBFs= 25, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.35217
RBFs= 25, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.38912
RBFs= 25, sigma=0.50, eta=0.500, epochs=1000, Val MAE=0.84189
RBFs= 30, sigma=0.50, eta=0.001, epochs= 283, Val MAE=0.36775
RBFs= 30, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.36581
RBFs= 30, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.35638
RBFs= 30, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.41485
RBFs= 30, sigma=0.50, eta=0.500, epochs=1000, Val MAE=20938771061807.91406
RBFs= 35, sigma=0.50, eta=0.001, epochs= 271, Val MAE=0.36958
RBFs= 35, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.36290
RBFs= 35, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.36179
RBFs= 35, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.44038
RBFs= 35, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 40, sigma=0.50, eta=0.001, epochs= 246, Val MAE=0.37006
RBFs= 40, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.36026
RBFs= 40, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.36803
RBFs= 40, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.46496
RBFs= 40, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 45, sigma=0.50, eta=0.001, epochs= 182, Val MAE=0.36694
RBFs= 45, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.35788
RBFs= 45, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.37493
RBFs= 45, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.48869
RBFs= 45, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 50, sigma=0.50, eta=0.001, epochs= 182, Val MAE=0.36856
RBFs= 50, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.35574
RBFs= 50, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.38640
RBFs= 50, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.51426
RBFs= 50, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs=100, sigma=0.50, eta=0.001, epochs=1000, Val MAE=0.37806
RBFs=100, sigma=0.50, eta=0.010, epochs=1000, Val MAE=0.34428
RBFs=100, sigma=0.50, eta=0.050, epochs=1000, Val MAE=0.51164
RBFs=100, sigma=0.50, eta=0.100, epochs=1000, Val MAE=0.68246
RBFs=100, sigma=0.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 10, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.43066
RBFs= 10, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.35813
RBFs= 10, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.39211
RBFs= 10, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.44442
RBFs= 10, sigma=1.00, eta=0.500, epochs=1000, Val MAE=0.69829
RBFs= 15, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.40358
RBFs= 15, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.35788
RBFs= 15, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.41734
RBFs= 15, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.50285
RBFs= 15, sigma=1.00, eta=0.500, epochs=1000, Val MAE=2.65660
RBFs= 20, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.39015
RBFs= 20, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.35912
RBFs= 20, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.44584
RBFs= 20, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.56082
RBFs= 20, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 25, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.37937
RBFs= 25, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.36125
RBFs= 25, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.47572
RBFs= 25, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.62516
RBFs= 25, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 30, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.37021
RBFs= 30, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.36561
RBFs= 30, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.50671
RBFs= 30, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.67136
RBFs= 30, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 35, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.36219
RBFs= 35, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.37029
RBFs= 35, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.53558
RBFs= 35, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.69995
RBFs= 35, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 40, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.35539
RBFs= 40, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.37525
RBFs= 40, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.56703
RBFs= 40, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.71256
RBFs= 40, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 45, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.35474
RBFs= 45, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.38045
RBFs= 45, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.60096
RBFs= 45, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.71334
RBFs= 45, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 50, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.35430
RBFs= 50, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.38586
RBFs= 50, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.63037
RBFs= 50, sigma=1.00, eta=0.100, epochs=1000, Val MAE=0.71177
RBFs= 50, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs=100, sigma=1.00, eta=0.001, epochs=1000, Val MAE=0.35271
RBFs=100, sigma=1.00, eta=0.010, epochs=1000, Val MAE=0.44583
RBFs=100, sigma=1.00, eta=0.050, epochs=1000, Val MAE=0.71483
RBFs=100, sigma=1.00, eta=0.100, epochs=1000, Val MAE=nan
RBFs=100, sigma=1.00, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 10, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.71233
RBFs= 10, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.61428
RBFs= 10, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.56423
RBFs= 10, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.65912
RBFs= 10, sigma=1.50, eta=0.500, epochs=1000, Val MAE=1.13413
RBFs= 15, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.67478
RBFs= 15, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.60095
RBFs= 15, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.60648
RBFs= 15, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.76483
RBFs= 15, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 20, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.65107
RBFs= 20, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.59260
RBFs= 20, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.68216
RBFs= 20, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.81534
RBFs= 20, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 25, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.63593
RBFs= 25, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.58652
RBFs= 25, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.74201
RBFs= 25, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.82576
RBFs= 25, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 30, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.62612
RBFs= 30, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.58173
RBFs= 30, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.78592
RBFs= 30, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.81587
RBFs= 30, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 35, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.61963
RBFs= 35, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.57773
RBFs= 35, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.81579
RBFs= 35, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.82143
RBFs= 35, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 40, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.61522
RBFs= 40, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.57466
RBFs= 40, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.83374
RBFs= 40, sigma=1.50, eta=0.100, epochs=1000, Val MAE=0.90583
RBFs= 40, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 45, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.61212
RBFs= 45, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.57872
RBFs= 45, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.84178
RBFs= 45, sigma=1.50, eta=0.100, epochs=1000, Val MAE=1.18175
RBFs= 45, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs= 50, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.60984
RBFs= 50, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.58301
RBFs= 50, sigma=1.50, eta=0.050, epochs=1000, Val MAE=0.84213
RBFs= 50, sigma=1.50, eta=0.100, epochs=1000, Val MAE=1.86496
RBFs= 50, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan
RBFs=100, sigma=1.50, eta=0.001, epochs=1000, Val MAE=0.59990
RBFs=100, sigma=1.50, eta=0.010, epochs=1000, Val MAE=0.70130
RBFs=100, sigma=1.50, eta=0.050, epochs=1000, Val MAE=2.00234
RBFs=100, sigma=1.50, eta=0.100, epochs=1000, Val MAE=nan
RBFs=100, sigma=1.50, eta=0.500, epochs=1000, Val MAE=nan

BEST CONFIGURATION
RBFs: 25
Sigma: 0.25
Eta: 0.001
Epochs: 1000
Validation MAE: 0.31885

FINAL TEST PERFORMANCE
Noisy test MAE: 0.35540
Clean test MAE: 0.25145

========== FINAL ONLINE SUMMARY ==========

SINE
RBFs: 15
Sigma: 1.0
Eta: 0.1
Epochs: 1000
Validation MAE: 0.17744
Noisy test MAE: 0.25127
Clean test MAE: 0.06503

SQUARE
RBFs: 25
Sigma: 0.25
Eta: 0.001
Epochs: 1000
Validation MAE: 0.31885
Noisy test MAE: 0.35540
Clean test MAE: 0.25145"""