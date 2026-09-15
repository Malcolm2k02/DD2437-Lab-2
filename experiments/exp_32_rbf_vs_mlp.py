from src.data_setup import *
from src.rbf import *
from src.mlp import MLP

import numpy as np
import time


# ============================================================
# Configuration
# ============================================================

sin_n_hidden = 10
sin_sigma = 1.5

square_n_hidden = 40
square_sigma = 0.25


# MLP
mlp_learning_rate = 0.05
mlp_epochs = 3000


# ============================================================
# Comparison function
# ============================================================

def compare_models(
    name,
    train_targets,
    noisy_test_targets,
    clean_test_targets,
    n_hidden,
    sigma
):

    print(f"\n========== {name} ==========")

    print(
        f"Hidden units: {n_hidden}, "
        f"RBF sigma: {sigma}"
    )

    # ========================================================
    # RBF
    # ========================================================

    mus = np.linspace(
        0,
        2 * np.pi,
        n_hidden
    )

    Phi_train = design_matrix(
        x_train,
        mus,
        sigma
    )

    start = time.perf_counter()

    rbf_weights = train_least_squares(
        Phi_train,
        train_targets
    )

    rbf_time = time.perf_counter() - start

    Phi_test = design_matrix(
        x_test,
        mus,
        sigma
    )

    rbf_predictions = Phi_test @ rbf_weights

    rbf_noisy_mae = residual_error(
        rbf_predictions,
        noisy_test_targets
    )

    rbf_clean_mae = residual_error(
        rbf_predictions,
        clean_test_targets
    )


    # ========================================================
    # MLP
    # ========================================================

    mlp = MLP(
        n_hidden=n_hidden,
        learning_rate=mlp_learning_rate,
        seed=42
    )

    start = time.perf_counter()

    mlp_errors = mlp.train(
        x_train,
        train_targets,
        epochs=mlp_epochs
    )

    mlp_time = time.perf_counter() - start

    mlp_predictions = mlp.predict(
        x_test
    )

    mlp_noisy_mae = residual_error(
        mlp_predictions,
        noisy_test_targets
    )

    mlp_clean_mae = residual_error(
        mlp_predictions,
        clean_test_targets
    )


    # ========================================================
    # Results
    # ========================================================

    print("\nRBF:")
    print(f"Noisy test MAE: {rbf_noisy_mae:.5f}")
    print(f"Clean test MAE: {rbf_clean_mae:.5f}")
    print(f"Training time: {rbf_time:.6f} seconds")

    print("\nMLP:")
    print(f"Noisy test MAE: {mlp_noisy_mae:.5f}")
    print(f"Clean test MAE: {mlp_clean_mae:.5f}")
    print(f"Training time: {mlp_time:.6f} seconds")
    print(f"Final training MSE: {mlp_errors[-1]:.5f}")

    return (
        rbf_predictions,
        mlp_predictions,
        mlp_errors
    )

# ============================================================
# Sine
# ============================================================

sin_rbf_pred, sin_mlp_pred, sin_mlp_errors = compare_models(
    name="NOISY SINE",
    train_targets=sin_train_noisy,
    noisy_test_targets=sin_test_noisy,
    clean_test_targets=sin_test,
    n_hidden=sin_n_hidden,
    sigma=sin_sigma
)


# ============================================================
# Square
# ============================================================

square_rbf_pred, square_mlp_pred, square_mlp_errors = compare_models(
    name="NOISY SQUARE",
    train_targets=square_train_noisy,
    noisy_test_targets=square_test_noisy,
    clean_test_targets=square_test,
    n_hidden=square_n_hidden,
    sigma=square_sigma
)

"""lr 0.01 for mlp========== NOISY SINE ==========

RBF:
Noisy test MAE: 0.27730
Clean test MAE: 0.14774
Training time: 0.104575 seconds

MLP:
Noisy test MAE: 0.62249
Clean test MAE: 0.56250
Training time: 0.560460 seconds
Final training MSE: 0.48873

========== NOISY SQUARE ==========

RBF:
Noisy test MAE: 0.34100
Clean test MAE: 0.23403
Training time: 0.000740 seconds

MLP:
Noisy test MAE: 0.81959
Clean test MAE: 0.84280
Training time: 0.356216 seconds
Final training MSE: 0.89259


lr 0.05 for mlp
========== NOISY SINE ==========

RBF:
Noisy test MAE: 0.27730
Clean test MAE: 0.14774
Training time: 0.050136 seconds

MLP:
Noisy test MAE: 0.33278
Clean test MAE: 0.23518
Training time: 0.489785 seconds
Final training MSE: 0.12931

========== NOISY SQUARE ==========

RBF:
Noisy test MAE: 0.34100
Clean test MAE: 0.23403
Training time: 0.000865 seconds

MLP:
Noisy test MAE: 0.46746
Clean test MAE: 0.34717
Training time: 0.507701 seconds
Final training MSE: 0.24565


The MLP was sensitive to weight initialization. 
Increasing the variance of the 
initial weights gave the hidden neurons
 more diverse initial responses and substantially improved learning."""



 """LATEST RESULTS
 ========== NOISY SINE ==========
Hidden units: 10, RBF sigma: 1.5

RBF:
Noisy test MAE: 0.25244
Clean test MAE: 0.11529
Training time: 0.084490 seconds

MLP:
Noisy test MAE: 0.36457
Clean test MAE: 0.28317
Training time: 0.416963 seconds
Final training MSE: 0.16553

========== NOISY SQUARE ==========
Hidden units: 40, RBF sigma: 0.25

RBF:
Noisy test MAE: 0.31556
Clean test MAE: 0.21169
Training time: 0.006160 seconds

MLP:
Noisy test MAE: 0.48227
Clean test MAE: 0.37416
Training time: 0.484572 seconds
Final training MSE: 0.27126"""