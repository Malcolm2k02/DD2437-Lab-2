import numpy as np
import matplotlib.pyplot as plt
from src.rbf import *
from src.data_setup import *

"""for n_rbf in [8, 9, 12, 20]:
    mus = np.linspace(0, 2 * np.pi, n_rbf)
    sigma = 0.5

    Phi_train = design_matrix(x_train, mus, sigma)
    # print("Design matrix shape:", Phi_train.shape)
    weights = train_least_squares(Phi_train, sin_train)
    #print(weights)
    Phi_test = design_matrix(x_test, mus, sigma)
    predictions = Phi_test @ weights
    error = residual_error(predictions, sin_test)
    
    weights_square = train_least_squares(Phi_train, square_train)
    predictions_square = Phi_test @ weights_square
    transformed_square = np.where(predictions_square >= 0, 1, -1)
    error_square = residual_error(predictions_square, square_test)
    transformed_error_square = residual_error(transformed_square, square_test)
    print(f"RBFs: {n_rbf}, Sin error: {error:.6f}, Square error: {error_square:.6f}, Transformed square error: {transformed_error_square:.6f}")

    #plot_rbf_results(x_test, sin_test, predictions, mus, title="RBF approximation of sin(2x)")
    #plot_rbf_results(x_test, square_test, predictions_square, mus, title="RBF approximation of square wave")
    #plot_rbf_results(x_test, square_test, transformed_square, mus, title="Transformed RBF approximation of square wave")
"""

thresholds = [0.1, 0.01, 0.001]

# Store the lowest number of RBFs reaching each threshold
sin_thresholds = {t: None for t in thresholds}
square_thresholds = {t: None for t in thresholds}
transformed_square_thresholds = {t: None for t in thresholds}

# Try increasing numbers of RBF nodes
for n_rbf in range(1, 100):
    mus = np.linspace(0, 2 * np.pi, n_rbf)
    sigma = 0.5

    Phi_train = design_matrix(x_train, mus, sigma)
    Phi_test = design_matrix(x_test, mus, sigma)

    # Sin
    weights = train_least_squares(Phi_train, sin_train)
    predictions = Phi_test @ weights
    error = residual_error(predictions, sin_test)

    # Square
    weights_square = train_least_squares(Phi_train, square_train)
    predictions_square = Phi_test @ weights_square
    error_square = residual_error(predictions_square, square_test)

    # Thresholded square
    transformed_square = np.where(predictions_square >= 0, 1, -1)
    transformed_error_square = residual_error(transformed_square, square_test)

    print(
        f"RBFs: {n_rbf:2d}, "
        f"Sin error: {error:.6f}, "
        f"Square error: {error_square:.6f}, "
        f"Transformed square error: {transformed_error_square:.6f}"
    )

    # Save the FIRST number of RBFs that reaches each threshold
    for threshold in thresholds:
        if sin_thresholds[threshold] is None and error < threshold:
            sin_thresholds[threshold] = n_rbf

        if square_thresholds[threshold] is None and error_square < threshold:
            square_thresholds[threshold] = n_rbf

        if (transformed_square_thresholds[threshold] is None
                and transformed_error_square < threshold):
            transformed_square_thresholds[threshold] = n_rbf


print("\n========== LOWEST NUMBER OF RBFs ==========")

for threshold in thresholds:
    print(f"\nError < {threshold}:")
    print(f"  Sin:                {sin_thresholds[threshold]}")
    print(f"  Square:             {square_thresholds[threshold]}")
    print(f"  Transformed square: {transformed_square_thresholds[threshold]}")

"""
Error < 0.1:
  Sin:                8
  Square:             None
  Transformed square: 6

Error < 0.01:
  Sin:                12
  Square:             None
  Transformed square: 9

Error < 0.001:
  Sin:                20
  Square:             None
  Transformed square: 9"""

import numpy as np
import matplotlib.pyplot as plt
from src.rbf import *
from src.data_setup import *


# ============================================================
# Helper function
# ============================================================

def get_prediction(n_rbf, target_train, sigma=0.5):
    mus = np.linspace(0, 2 * np.pi, n_rbf)

    Phi_train = design_matrix(x_train, mus, sigma)
    Phi_test = design_matrix(x_test, mus, sigma)

    weights = train_least_squares(Phi_train, target_train)
    predictions = Phi_test @ weights

    return predictions


# ============================================================
# 1. SIN(2x)
# Thresholds:
# < 0.1   -> 8 RBFs
# < 0.01  -> 12 RBFs
# < 0.001 -> 20 RBFs
# ============================================================

plt.figure(figsize=(8, 5))

# Target
plt.plot(
    x_test,
    sin_test,
    linewidth=2,
    label="Target sin(2x)"
)

# RBF predictions
for n_rbf, threshold in [(8, 0.1), (12, 0.01), (20, 0.001)]:

    predictions = get_prediction(n_rbf, sin_train)

    error = residual_error(predictions, sin_test)

    plt.plot(
        x_test,
        predictions,
        "--",
        label=f"{n_rbf} RBFs (MAE={error:.4f}, <{threshold})"
    )

plt.xlabel("x")
plt.ylabel("Output")
plt.title("RBF approximation of sin(2x)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("sin_threshold_comparison.png", dpi=300)
plt.show()


# ============================================================
# 2. SQUARE(2x)
#
# None of the tested RBF numbers reached < 0.1,
# so show several representative RBF sizes instead.
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    x_test,
    square_test,
    linewidth=2,
    label="Target square(2x)"
)

for n_rbf in [8, 12, 20]:

    predictions = get_prediction(n_rbf, square_train)

    error = residual_error(predictions, square_test)

    plt.plot(
        x_test,
        predictions,
        "--",
        label=f"{n_rbf} RBFs (MAE={error:.4f})"
    )

plt.xlabel("x")
plt.ylabel("Output")
plt.title("RBF approximation of square(2x)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("square_threshold_comparison.png", dpi=300)
plt.show()


# ============================================================
# 3. THRESHOLDED SQUARE(2x)
# Thresholds:
# < 0.1   -> 6 RBFs
# < 0.01  -> 9 RBFs
# < 0.001 -> 9 RBFs
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    x_test,
    square_test,
    linewidth=2,
    label="Target square(2x)"
)

# 6 RBFs
predictions_6 = get_prediction(6, square_train)
transformed_6 = np.where(predictions_6 >= 0, 1, -1)

error_6 = residual_error(
    transformed_6,
    square_test
)

plt.plot(
    x_test,
    transformed_6,
    "--",
    label=f"6 RBFs (MAE={error_6:.4f}, <0.1)"
)


# 9 RBFs
predictions_9 = get_prediction(9, square_train)
transformed_9 = np.where(predictions_9 >= 0, 1, -1)

error_9 = residual_error(
    transformed_9,
    square_test
)

plt.plot(
    x_test,
    transformed_9,
    ":",
    linewidth=2,
    label=f"9 RBFs (MAE={error_9:.4f}, <0.01 and <0.001)"
)

plt.xlabel("x")
plt.ylabel("Output")
plt.title("Thresholded RBF approximation of square(2x)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("transformed_square_threshold_comparison.png", dpi=300)
plt.show()