from src.data_setup import *
from src.rbf import *
from src.mlp import MLP

import numpy as np
import time


# ============================================================
# Configuration
# ============================================================

n_hidden = 50

# RBF configuration
sigma = 1.0
mus = np.linspace(0, 2 * np.pi, n_hidden)

# MLP configuration
mlp_eta = 0.01
mlp_epochs = 3000

def compare_models(
    name,
    train_targets,
    noisy_test_targets,
    clean_test_targets
):

    print(f"\n========== {name} ==========")


    # ========================================================
    # RBF
    # ========================================================

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


    rbf_noisy_error = residual_error(
        rbf_predictions,
        noisy_test_targets
    )

    rbf_clean_error = residual_error(
        rbf_predictions,
        clean_test_targets
    )


    # ========================================================
    # MLP
    # ========================================================

    mlp = MLP(
        n_hidden=n_hidden,
        eta=mlp_eta,
        seed=42
    )

    start = time.perf_counter()

    mlp.train(
        x_train,
        train_targets,
        epochs=mlp_epochs
    )

    mlp_time = time.perf_counter() - start


    mlp_predictions = mlp.predict(x_test)


    mlp_noisy_error = residual_error(
        mlp_predictions,
        noisy_test_targets
    )

    mlp_clean_error = residual_error(
        mlp_predictions,
        clean_test_targets
    )


    # ========================================================
    # Results
    # ========================================================

    print("\nRBF:")
    print(f"Noisy test MAE: {rbf_noisy_error:.5f}")
    print(f"Clean test MAE: {rbf_clean_error:.5f}")
    print(f"Training time: {rbf_time:.6f} seconds")

    print("\nMLP:")
    print(f"Noisy test MAE: {mlp_noisy_error:.5f}")
    print(f"Clean test MAE: {mlp_clean_error:.5f}")
    print(f"Training time: {mlp_time:.6f} seconds")

    return rbf_predictions, mlp_predictions

sin_rbf_pred, sin_mlp_pred = compare_models(
    name="NOISY SINE",
    train_targets=sin_train_noisy,
    noisy_test_targets=sin_test_noisy,
    clean_test_targets=sin_test
)

square_rbf_pred, square_mlp_pred = compare_models(
    name="NOISY SQUARE",
    train_targets=square_train_noisy,
    noisy_test_targets=square_test_noisy,
    clean_test_targets=square_test
)