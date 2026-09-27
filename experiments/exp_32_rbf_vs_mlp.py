import time
import numpy as np
import matplotlib.pyplot as plt

from src.mlp import MLP
from src.data_setup import *
from src.rbf import *


# ============================================================
# Train and evaluate MLP
# ============================================================

def evaluate_mlp(
    function_name,
    x_train,
    y_train,
    x_test,
    y_test_noisy,
    y_test_clean,
    n_hidden,
    epochs=3000
):

    mlp = MLP(
        n_hidden=n_hidden,
        learning_rate=0.01,
        alpha=0.9,
        seed=42
    )

    start = time.perf_counter()

    errors = mlp.train(
        x_train,
        y_train,
        epochs=epochs
    )

    training_time = time.perf_counter() - start

    predictions = mlp.predict(x_test)

    noisy_test_mae = residual_error(
        predictions,
        y_test_noisy
    )

    clean_test_mae = residual_error(
        predictions,
        y_test_clean
    )

    print(f"\n========== MLP - {function_name.upper()} ==========")
    print(f"Hidden units:     {n_hidden}")
    print(f"Epochs:           {epochs}")
    print(f"Training time:    {training_time:.6f} s")
    print(f"Noisy test MAE:   {noisy_test_mae:.5f}")
    print(f"Clean test MAE:   {clean_test_mae:.5f}")
    print(f"Final train MSE:  {errors[-1]:.5f}")

    return {
        "predictions": predictions,
        "training_time": training_time,
        "noisy_test_mae": noisy_test_mae,
        "clean_test_mae": clean_test_mae,
        "train_mse": errors[-1]
    }


# ============================================================
# Train and evaluate batch RBF
# ============================================================

def evaluate_rbf(
    function_name,
    x_train,
    y_train,
    x_test,
    y_test_noisy,
    y_test_clean,
    n_rbf,
    sigma
):

    centers = np.linspace(
        0,
        2 * np.pi,
        n_rbf
    )

    start = time.perf_counter()

    Phi_train = design_matrix(
        x_train,
        centers,
        sigma
    )

    weights = train_least_squares(
        Phi_train,
        y_train
    )

    training_time = time.perf_counter() - start

    Phi_test = design_matrix(
        x_test,
        centers,
        sigma
    )

    predictions = Phi_test @ weights

    noisy_test_mae = residual_error(
        predictions,
        y_test_noisy
    )

    clean_test_mae = residual_error(
        predictions,
        y_test_clean
    )

    print(f"\n========== RBF - {function_name.upper()} ==========")
    print(f"RBF units:        {n_rbf}")
    print(f"Sigma:            {sigma}")
    print(f"Training time:    {training_time:.6f} s")
    print(f"Noisy test MAE:   {noisy_test_mae:.5f}")
    print(f"Clean test MAE:   {clean_test_mae:.5f}")

    return {
        "predictions": predictions,
        "training_time": training_time,
        "noisy_test_mae": noisy_test_mae,
        "clean_test_mae": clean_test_mae
    }


# ============================================================
# Plot comparison
# ============================================================

def plot_comparison(
    x_test,
    y_clean,
    y_noisy,
    rbf_predictions,
    mlp_predictions,
    function_name
):

    # Sort x so the prediction lines are drawn correctly
    order = np.argsort(x_test)

    x = x_test[order]
    clean = y_clean[order]
    noisy = y_noisy[order]
    rbf = rbf_predictions[order]
    mlp = mlp_predictions[order]

    plt.figure(figsize=(9, 5))

    # Original clean function
    plt.plot(
        x,
        clean,
        linewidth=2,
        label="Clean target"
    )

    # Noisy observations
    plt.scatter(
        x,
        noisy,
        s=18,
        alpha=0.5,
        label="Noisy test data"
    )

    # RBF approximation
    plt.plot(
        x,
        rbf,
        "--",
        linewidth=2,
        label="RBF"
    )

    # MLP approximation
    plt.plot(
        x,
        mlp,
        "-.",
        linewidth=2,
        label="MLP"
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.title(
        f"RBF vs MLP — Noisy {function_name}"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"rbf_vs_mlp_{function_name.lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# SINE
# 10 units for BOTH networks
# Best RBF: sigma = 1.0
# ============================================================

sine_rbf = evaluate_rbf(
    "Sine",
    x_train,
    sin_train_noisy,
    x_test,
    sin_test_noisy,
    sin_test,
    n_rbf=10,
    sigma=1.0
)

sine_mlp = evaluate_mlp(
    "Sine",
    x_train,
    sin_train_noisy,
    x_test,
    sin_test_noisy,
    sin_test,
    n_hidden=10
)

plot_comparison(
    x_test,
    sin_test,
    sin_test_noisy,
    sine_rbf["predictions"],
    sine_mlp["predictions"],
    "Sine"
)


# ============================================================
# SQUARE
# 20 units for BOTH networks
# Best RBF: sigma = 0.5
# ============================================================

square_rbf = evaluate_rbf(
    "Square",
    x_train,
    square_train_noisy,
    x_test,
    square_test_noisy,
    square_test,
    n_rbf=20,
    sigma=0.5
)

square_mlp = evaluate_mlp(
    "Square",
    x_train,
    square_train_noisy,
    x_test,
    square_test_noisy,
    square_test,
    n_hidden=20
)

plot_comparison(
    x_test,
    square_test,
    square_test_noisy,
    square_rbf["predictions"],
    square_mlp["predictions"],
    "Square"
)


# ============================================================
# Final summary
# ============================================================

print("\n========== FINAL COMPARISON ==========")

print("\nSINE")
print(
    f"RBF | MAE: {sine_rbf['noisy_test_mae']:.5f} | "
    f"Time: {sine_rbf['training_time']:.6f} s"
)
print(
    f"MLP | MAE: {sine_mlp['noisy_test_mae']:.5f} | "
    f"Time: {sine_mlp['training_time']:.6f} s"
)

print("\nSQUARE")
print(
    f"RBF | MAE: {square_rbf['noisy_test_mae']:.5f} | "
    f"Time: {square_rbf['training_time']:.6f} s"
)
print(
    f"MLP | MAE: {square_mlp['noisy_test_mae']:.5f} | "
    f"Time: {square_mlp['training_time']:.6f} s"
)
