import numpy as np
import matplotlib.pyplot as plt

from exp3_3.competitive_learning import competitive_learning

# ============================================================
# Load ballistic data
# ============================================================

train_data = np.loadtxt("data/ballist.dat")
test_data = np.loadtxt("data/balltest.dat")

# First two columns: angle, velocity
X_train = train_data[:, :2]
X_test = test_data[:, :2]

# Last two columns: distance, height
Y_train = train_data[:, 2:]
Y_test = test_data[:, 2:]

print("Training input shape:", X_train.shape)
print("Training output shape:", Y_train.shape)

print("Test input shape:", X_test.shape)
print("Test output shape:", Y_test.shape)


# ============================================================
# Gaussian RBF
# ============================================================

def gaussian_rbf(x, center, sigma):

    distance_squared = np.sum(
        (x - center) ** 2
    )

    return np.exp(
        -distance_squared /
        (2 * sigma ** 2)
    )


# ============================================================
# Calculate sigma for each center
#
# Sigma = distance to nearest neighbouring center
# ============================================================

def calculate_sigmas_2d(centers):

    n_centers = len(centers)

    sigmas = np.zeros(n_centers)

    for i in range(n_centers):

        distances = np.linalg.norm(
            centers - centers[i],
            axis=1
        )

        # Ignore distance from center to itself
        distances[i] = np.inf

        sigmas[i] = np.min(distances)

    return sigmas


# ============================================================
# Design matrix
# ============================================================

def design_matrix_2d(X, centers, sigmas):

    Phi = np.zeros(
        (len(X), len(centers))
    )

    for i, x in enumerate(X):

        for j, center in enumerate(centers):

            Phi[i, j] = gaussian_rbf(
                x,
                center,
                sigmas[j]
            )

    return Phi


# ============================================================
# Delta learning for TWO outputs
#
# weights shape:
#
#     n_units x 2
#
# Column 0 -> distance
# Column 1 -> height
# ============================================================

def train_delta_2d(
    X,
    targets,
    centers,
    sigmas,
    eta=0.01,
    epochs=1000,
    seed=42
):

    rng = np.random.default_rng(seed)

    n_units = len(centers)

    weights = np.zeros(
        (n_units, 2)
    )

    Phi = design_matrix_2d(
        X,
        centers,
        sigmas
    )

    for epoch in range(epochs):

        order = rng.permutation(
            len(X)
        )

        for i in order:

            phi = Phi[i]

            prediction = (
                phi @ weights
            )

            error = (
                targets[i] -
                prediction
            )

            # Outer product:
            #
            # phi:  (n_units,)
            # error: (2,)
            #
            # update: (n_units, 2)

            weights += (
                eta *
                np.outer(phi, error)
            )

    return weights


# ============================================================
# Settings
#
# Start with 15 RBF units.
# We will test several values afterwards.
# ============================================================

n_units = 40

eta_cl = 0.1
cl_epochs = 100

eta_delta = 0.01
delta_epochs = 1000

seed = 42


# ============================================================
# Competitive learning
# ============================================================
if __name__ == "__main__":
    centers, wins = competitive_learning(
        X_train,
        n_units=n_units,
        eta=eta_cl,
        epochs=cl_epochs,
        seed=seed
    )


    # ============================================================
    # Determine RBF widths
    # ============================================================

    sigmas = calculate_sigmas_2d(
        centers
    )


    # ============================================================
    # Train output weights
    # ============================================================

    weights = train_delta_2d(
        X_train,
        Y_train,
        centers,
        sigmas,
        eta=eta_delta,
        epochs=delta_epochs,
        seed=seed
    )


    # ============================================================
    # Predictions
    # ============================================================

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

    Y_train_pred = (
        Phi_train @ weights
    )

    Y_test_pred = (
        Phi_test @ weights
    )


    # ============================================================
    # Errors
    # ============================================================

    train_distance_mae = np.mean(
        np.abs(
            Y_train_pred[:, 0] -
            Y_train[:, 0]
        )
    )

    train_height_mae = np.mean(
        np.abs(
            Y_train_pred[:, 1] -
            Y_train[:, 1]
        )
    )

    test_distance_mae = np.mean(
        np.abs(
            Y_test_pred[:, 0] -
            Y_test[:, 0]
        )
    )

    test_height_mae = np.mean(
        np.abs(
            Y_test_pred[:, 1] -
            Y_test[:, 1]
        )
    )

    overall_train_mae = np.mean(
        np.abs(
            Y_train_pred -
            Y_train
        )
    )

    overall_test_mae = np.mean(
        np.abs(
            Y_test_pred -
            Y_test
        )
    )


    # ============================================================
    # Dead units
    # ============================================================

    dead_units = np.sum(
        wins == 0
    )


    # ============================================================
    # Results
    # ============================================================

    print("\n")
    print("=" * 60)
    print("BALLISTIC RBF WITH COMPETITIVE LEARNING")
    print("=" * 60)

    print(f"\nNumber of RBF units: {n_units}")

    print("\nTRAINING ERROR")

    print(
        f"Distance MAE: {train_distance_mae:.5f}"
    )

    print(
        f"Height MAE:   {train_height_mae:.5f}"
    )

    print(
        f"Overall MAE:  {overall_train_mae:.5f}"
    )


    print("\nTEST ERROR")

    print(
        f"Distance MAE: {test_distance_mae:.5f}"
    )

    print(
        f"Height MAE:   {test_height_mae:.5f}"
    )

    print(
        f"Overall MAE:  {overall_test_mae:.5f}"
    )


    print("\nCOMPETITIVE LEARNING")

    print(
        f"Dead units: {dead_units}/{n_units}"
    )

    print("\nWins:")
    print(wins)

    print("\nCenters:")
    print(centers)

    print("\nSigmas:")
    print(sigmas)


    # ============================================================
    # Plot 1:
    # Training inputs and learned RBF centers
    # ============================================================

    plt.figure(figsize=(7, 6))

    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        alpha=0.5,
        label="Training samples"
    )

    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="x",
        s=100,
        linewidths=2,
        label="RBF centers"
    )

    plt.xlabel("Angle")
    plt.ylabel("Velocity")

    plt.title(
        "Ballistic input space and CL-positioned RBF units"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


    # ============================================================
    # Plot 2:
    # True vs predicted distance
    # ============================================================

    plt.figure(figsize=(6, 6))

    plt.scatter(
        Y_test[:, 0],
        Y_test_pred[:, 0]
    )

    minimum = min(
        Y_test[:, 0].min(),
        Y_test_pred[:, 0].min()
    )

    maximum = max(
        Y_test[:, 0].max(),
        Y_test_pred[:, 0].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--"
    )

    plt.xlabel("True distance")
    plt.ylabel("Predicted distance")

    plt.title(
        "Ballistic test set — distance prediction"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.show()


    # ============================================================
    # Plot 3:
    # True vs predicted height
    # ============================================================

    plt.figure(figsize=(6, 6))

    plt.scatter(
        Y_test[:, 1],
        Y_test_pred[:, 1]
    )

    minimum = min(
        Y_test[:, 1].min(),
        Y_test_pred[:, 1].min()
    )

    maximum = max(
        Y_test[:, 1].max(),
        Y_test_pred[:, 1].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--"
    )

    plt.xlabel("True height")
    plt.ylabel("Predicted height")

    plt.title(
        "Ballistic test set — height prediction"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.show()