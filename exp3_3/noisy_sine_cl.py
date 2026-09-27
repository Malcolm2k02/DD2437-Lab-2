import numpy as np
import matplotlib.pyplot as plt

from src.data_setup import *
from src.rbf import (
    residual_error,
    train_delta_variable_sigma,
    design_matrix_variable_sigma
)
from exp3_3.competitive_learning import (
    competitive_learning,
    calculate_sigmas
)


# ============================================================
# Settings
# ============================================================

# Best online/delta architecture from section 3.2
n_units = 15

eta_cl = 0.1
eta_delta = 0.1

cl_epochs = 100
delta_epochs = 1000

n_runs = 10


# ============================================================
# Store results
# ============================================================

cl_noisy_maes = []
cl_clean_maes = []
cl_epochs_used = []

cl_centers_all = []
cl_sigmas_all = []
cl_wins_all = []

manual_noisy_maes = []
manual_clean_maes = []
manual_epochs_used = []


# ============================================================
# Manual centers
# ============================================================

centers_manual = np.linspace(
    0,
    2 * np.pi,
    n_units
)

sigmas_manual = calculate_sigmas(
    centers_manual
)


# ============================================================
# Run experiment 10 times
# ============================================================

for run in range(n_runs):

    seed = 42 + run

    print(
        f"\n================ RUN {run + 1}/{n_runs} ================"
    )


    # ========================================================
    # CL RBF
    # ========================================================

    centers_cl, wins = competitive_learning(
        x_train,
        n_units=n_units,
        eta=eta_cl,
        epochs=cl_epochs,
        seed=seed
    )

    centers_cl = centers_cl.flatten()


    # --------------------------------------------------------
    # Sort centers and corresponding win counts
    # --------------------------------------------------------

    sort_idx = np.argsort(centers_cl)

    centers_cl = centers_cl[sort_idx]
    wins = wins[sort_idx]


    # --------------------------------------------------------
    # Determine widths from learned center positions
    # --------------------------------------------------------

    sigmas_cl = calculate_sigmas(
        centers_cl
    )


    # --------------------------------------------------------
    # Train output weights with delta learning
    #
    # IMPORTANT:
    # Training targets are NOISY.
    # --------------------------------------------------------

    np.random.seed(seed)

    weights_cl, epochs_cl = train_delta_variable_sigma(
        x_train,
        sin_train_noisy,
        centers_cl,
        sigmas_cl,
        eta=eta_delta,
        max_epochs=delta_epochs
    )


    # --------------------------------------------------------
    # CL test predictions
    # --------------------------------------------------------

    Phi_test_cl = design_matrix_variable_sigma(
        x_test,
        centers_cl,
        sigmas_cl
    )

    predictions_cl = (
        Phi_test_cl @ weights_cl
    )


    # --------------------------------------------------------
    # Evaluate against noisy AND clean test targets
    # --------------------------------------------------------

    cl_noisy_mae = residual_error(
        predictions_cl,
        sin_test_noisy
    )

    cl_clean_mae = residual_error(
        predictions_cl,
        sin_test
    )


    # --------------------------------------------------------
    # Store CL results
    # --------------------------------------------------------

    cl_noisy_maes.append(
        cl_noisy_mae
    )

    cl_clean_maes.append(
        cl_clean_mae
    )

    cl_epochs_used.append(
        epochs_cl
    )

    cl_centers_all.append(
        centers_cl.copy()
    )

    cl_sigmas_all.append(
        sigmas_cl.copy()
    )

    cl_wins_all.append(
        wins.copy()
    )


    # ========================================================
    # MANUALLY POSITIONED RBF
    # ========================================================

    np.random.seed(seed)

    weights_manual, epochs_manual = train_delta_variable_sigma(
        x_train,
        sin_train_noisy,
        centers_manual,
        sigmas_manual,
        eta=eta_delta,
        max_epochs=delta_epochs
    )


    # --------------------------------------------------------
    # Manual test predictions
    # --------------------------------------------------------

    Phi_test_manual = design_matrix_variable_sigma(
        x_test,
        centers_manual,
        sigmas_manual
    )

    predictions_manual = (
        Phi_test_manual @ weights_manual
    )


    # --------------------------------------------------------
    # Evaluate against noisy AND clean targets
    # --------------------------------------------------------

    manual_noisy_mae = residual_error(
        predictions_manual,
        sin_test_noisy
    )

    manual_clean_mae = residual_error(
        predictions_manual,
        sin_test
    )


    # --------------------------------------------------------
    # Store manual results
    # --------------------------------------------------------

    manual_noisy_maes.append(
        manual_noisy_mae
    )

    manual_clean_maes.append(
        manual_clean_mae
    )

    manual_epochs_used.append(
        epochs_manual
    )


    # ========================================================
    # Print current run
    # ========================================================

    print(
        f"CL     | Noisy MAE: {cl_noisy_mae:.5f} "
        f"| Clean MAE: {cl_clean_mae:.5f} "
        f"| epochs: {epochs_cl}"
    )

    print(
        f"Manual | Noisy MAE: {manual_noisy_mae:.5f} "
        f"| Clean MAE: {manual_clean_mae:.5f} "
        f"| epochs: {epochs_manual}"
    )


# ============================================================
# Convert results to numpy arrays
# ============================================================

cl_noisy_maes = np.array(
    cl_noisy_maes
)

cl_clean_maes = np.array(
    cl_clean_maes
)

cl_epochs_used = np.array(
    cl_epochs_used
)

manual_noisy_maes = np.array(
    manual_noisy_maes
)

manual_clean_maes = np.array(
    manual_clean_maes
)

manual_epochs_used = np.array(
    manual_epochs_used
)


# ============================================================
# Dead units
# ============================================================

dead_units_per_run = [
    np.sum(wins == 0)
    for wins in cl_wins_all
]


# ============================================================
# Final statistical results
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RESULTS - NOISY SINE")
print("=" * 60)


print("\nCL POSITIONING")

print(
    f"Noisy test MAE: "
    f"{np.mean(cl_noisy_maes):.5f} "
    f"+/- {np.std(cl_noisy_maes):.5f}"
)

print(
    f"Clean test MAE: "
    f"{np.mean(cl_clean_maes):.5f} "
    f"+/- {np.std(cl_clean_maes):.5f}"
)

print(
    f"Epochs: "
    f"{np.mean(cl_epochs_used):.1f} "
    f"+/- {np.std(cl_epochs_used):.1f}"
)


print("\nMANUAL POSITIONING")

print(
    f"Noisy test MAE: "
    f"{np.mean(manual_noisy_maes):.5f} "
    f"+/- {np.std(manual_noisy_maes):.5f}"
)

print(
    f"Clean test MAE: "
    f"{np.mean(manual_clean_maes):.5f} "
    f"+/- {np.std(manual_clean_maes):.5f}"
)

print(
    f"Epochs: "
    f"{np.mean(manual_epochs_used):.1f} "
    f"+/- {np.std(manual_epochs_used):.1f}"
)


print("\nCL DEAD UNITS")

print(
    f"Mean dead units: "
    f"{np.mean(dead_units_per_run):.2f}"
)

print(
    f"Dead units per run: "
    f"{dead_units_per_run}"
)


# ============================================================
# Choose representative CL run
#
# Select run with noisy MAE closest to mean noisy MAE.
# ============================================================

mean_cl_mae = np.mean(
    cl_noisy_maes
)

representative_run = np.argmin(
    np.abs(
        cl_noisy_maes - mean_cl_mae
    )
)

centers_cl = cl_centers_all[
    representative_run
]

sigmas_cl = cl_sigmas_all[
    representative_run
]


# ============================================================
# Re-train representative CL model for plotting
# ============================================================

seed = 42 + representative_run

np.random.seed(seed)

weights_cl, _ = train_delta_variable_sigma(
    x_train,
    sin_train_noisy,
    centers_cl,
    sigmas_cl,
    eta=eta_delta,
    max_epochs=delta_epochs
)

Phi_test_cl = design_matrix_variable_sigma(
    x_test,
    centers_cl,
    sigmas_cl
)

predictions_cl = (
    Phi_test_cl @ weights_cl
)


# ============================================================
# Re-train corresponding manual model for plotting
# ============================================================

np.random.seed(seed)

weights_manual, _ = train_delta_variable_sigma(
    x_train,
    sin_train_noisy,
    centers_manual,
    sigmas_manual,
    eta=eta_delta,
    max_epochs=delta_epochs
)

Phi_test_manual = design_matrix_variable_sigma(
    x_test,
    centers_manual,
    sigmas_manual
)

predictions_manual = (
    Phi_test_manual @ weights_manual
)


# ============================================================
# Print representative center information
# ============================================================

print(
    f"\nRepresentative CL run: "
    f"{representative_run + 1}"
)

print("\nCL centers:")
print(centers_cl)

print("\nCL sigmas:")
print(sigmas_cl)

print("\nManual centers:")
print(centers_manual)


# ============================================================
# Plot
# ============================================================

order = np.argsort(x_test)

plt.figure(figsize=(10, 5))


# ------------------------------------------------------------
# Clean underlying sine function
# ------------------------------------------------------------

plt.plot(
    x_test[order],
    sin_test[order],
    label="Clean target",
    linewidth=2
)


# ------------------------------------------------------------
# Noisy test observations
# ------------------------------------------------------------

plt.scatter(
    x_test,
    sin_test_noisy,
    s=18,
    alpha=0.4,
    label="Noisy test data"
)


# ------------------------------------------------------------
# Manual RBF prediction
# ------------------------------------------------------------

plt.plot(
    x_test[order],
    predictions_manual[order],
    "--",
    linewidth=2,
    label="Manual RBF"
)


# ------------------------------------------------------------
# CL RBF prediction
# ------------------------------------------------------------

plt.plot(
    x_test[order],
    predictions_cl[order],
    "-.",
    linewidth=2,
    label="CL RBF"
)


# ------------------------------------------------------------
# Center positions
# ------------------------------------------------------------

plt.scatter(
    centers_manual,
    np.full_like(
        centers_manual,
        -1.45
    ),
    marker="o",
    label="Manual centers"
)

plt.scatter(
    centers_cl,
    np.full_like(
        centers_cl,
        -1.35
    ),
    marker="x",
    s=70,
    label="CL centers"
)


plt.xlabel("x")
plt.ylabel("sin(2x)")

plt.title(
    "Manual vs CL RBF positioning — noisy sine"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()