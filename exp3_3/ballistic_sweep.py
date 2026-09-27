import numpy as np
from exp3_3.competitive_learning import competitive_learning
from exp3_3.ballistic_cl import calculate_sigmas_2d, train_delta_2d, design_matrix_2d
# Use the same functions as in ballistic_cl.py:
# competitive_learning
# calculate_sigmas_2d
# design_matrix_2d
# train_delta_2d

train_data = np.loadtxt("data/ballist.dat")
test_data = np.loadtxt("data/balltest.dat")

X_train = train_data[:, :2]
Y_train = train_data[:, 2:]

X_test = test_data[:, :2]
Y_test = test_data[:, 2:]


# ============================================================
# Settings
# ============================================================

n_units_values = [5, 10, 15, 20, 25, 30, 40]
n_runs = 10

eta_cl = 0.1
cl_epochs = 100

eta_delta = 0.01
delta_epochs = 1000


# ============================================================
# Architecture sweep
# ============================================================

results = []

for n_units in n_units_values:

    train_maes = []
    test_maes = []
    distance_maes = []
    height_maes = []
    dead_units_list = []

    for run in range(n_runs):

        seed = 42 + run

        # ----------------------------------------------------
        # 1. Position RBF centers using competitive learning
        # ----------------------------------------------------

        centers, wins = competitive_learning(
            X_train,
            n_units=n_units,
            eta=eta_cl,
            epochs=cl_epochs,
            seed=seed
        )

        # ----------------------------------------------------
        # 2. Determine width of each RBF
        # ----------------------------------------------------

        sigmas = calculate_sigmas_2d(centers)

        # ----------------------------------------------------
        # 3. Train output weights using delta learning
        # ----------------------------------------------------

        weights = train_delta_2d(
            X_train,
            Y_train,
            centers,
            sigmas,
            eta=eta_delta,
            epochs=delta_epochs,
            seed=seed
        )

        # ----------------------------------------------------
        # 4. Predictions
        # ----------------------------------------------------

        Phi_train = design_matrix_2d(
            X_train,
            centers,
            sigmas
        )

        Phi_test = design_matrix_2d(
            X_test,
            centers,
            sigmas
        )

        Y_train_pred = Phi_train @ weights
        Y_test_pred = Phi_test @ weights

        # ----------------------------------------------------
        # 5. Errors
        # ----------------------------------------------------

        train_mae = np.mean(
            np.abs(Y_train_pred - Y_train)
        )

        test_mae = np.mean(
            np.abs(Y_test_pred - Y_test)
        )

        distance_mae = np.mean(
            np.abs(
                Y_test_pred[:, 0] -
                Y_test[:, 0]
            )
        )

        height_mae = np.mean(
            np.abs(
                Y_test_pred[:, 1] -
                Y_test[:, 1]
            )
        )

        dead_units = np.sum(wins == 0)

        train_maes.append(train_mae)
        test_maes.append(test_mae)
        distance_maes.append(distance_mae)
        height_maes.append(height_mae)
        dead_units_list.append(dead_units)

    # ========================================================
    # Mean and standard deviation over 10 runs
    # ========================================================

    result = {
        "units": n_units,

        "train_mean": np.mean(train_maes),
        "train_std": np.std(train_maes),

        "test_mean": np.mean(test_maes),
        "test_std": np.std(test_maes),

        "distance_mean": np.mean(distance_maes),
        "distance_std": np.std(distance_maes),

        "height_mean": np.mean(height_maes),
        "height_std": np.std(height_maes),

        "dead_mean": np.mean(dead_units_list),
        "dead_std": np.std(dead_units_list)
    }

    results.append(result)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 95)
print("BALLISTIC RBF ARCHITECTURE SWEEP")
print("=" * 95)

print(
    f"{'Units':<8}"
    f"{'Train MAE':<22}"
    f"{'Test MAE':<22}"
    f"{'Distance MAE':<22}"
    f"{'Height MAE':<22}"
    f"{'Dead':<15}"
)

print("-" * 110)

for r in results:

    print(
        f"{r['units']:<8}"
        f"{r['train_mean']:.5f} +/- {r['train_std']:.5f}    "
        f"{r['test_mean']:.5f} +/- {r['test_std']:.5f}    "
        f"{r['distance_mean']:.5f} +/- {r['distance_std']:.5f}    "
        f"{r['height_mean']:.5f} +/- {r['height_std']:.5f}    "
        f"{r['dead_mean']:.2f} +/- {r['dead_std']:.2f}"
    )


# ============================================================
# Find architecture with lowest mean test MAE
# ============================================================

best = min(
    results,
    key=lambda r: r["test_mean"]
)

print("\n" + "=" * 60)
print("LOWEST MEAN TEST ERROR")
print("=" * 60)

print(f"Units: {best['units']}")

print(
    f"Train MAE: "
    f"{best['train_mean']:.5f} +/- "
    f"{best['train_std']:.5f}"
)

print(
    f"Test MAE: "
    f"{best['test_mean']:.5f} +/- "
    f"{best['test_std']:.5f}"
)

print(
    f"Distance MAE: "
    f"{best['distance_mean']:.5f} +/- "
    f"{best['distance_std']:.5f}"
)

print(
    f"Height MAE: "
    f"{best['height_mean']:.5f} +/- "
    f"{best['height_std']:.5f}"
)

print(
    f"Dead units: "
    f"{best['dead_mean']:.2f} +/- "
    f"{best['dead_std']:.2f}"
)