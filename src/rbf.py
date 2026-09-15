import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, mu, sigma):
    """Compute the Gaussian RBF function."""
    transformed = np.exp(-np.linalg.norm(x - mu) ** 2 / (2 * sigma **2))
    return transformed


def design_matrix(X, mus, sigma):
    """
    Args:
        X: Input data of shape (n_samples, n_features)
        mu: Centers of the RBFs of shape (n_rbf, n_features)
        sigma: Width of the RBFs (scalar)"""
    Phi = np.zeros((len(X), len(mus)))

    for i, x in enumerate(X):
        for j, mu in enumerate(mus):
            Phi[i, j] = gaussian_rbf(x, mu, sigma)

    return Phi


def train_least_squares(Phi, targets):
    """Train the RBF network using least squares."""
    w,_, _, _ = np.linalg.lstsq(Phi, targets, rcond=None)
    return w

def residual_error(predictions, targets):
    """Compute the residual error."""
    return np.mean(np.abs((predictions - targets)))


def predict_rbf(X, centers, sigma, weights):
    """Predict the output of the RBF network."""
    return ""


def train_delta(
    X,
    targets,
    mus,
    sigma,
    eta,
    max_epochs=1000,
    tolerance=1e-5,
    patience=5
):
    weights = np.zeros(len(mus))

    previous_error = float("inf")
    stable_epochs = 0

    for epoch in range(max_epochs):

        # Shuffle training samples
        indices = np.random.permutation(len(X))

        for i in indices:
            x = X[i]
            target = targets[i]

            phi = np.array([
                gaussian_rbf(x, mu, sigma)
                for mu in mus
            ])

            prediction = phi @ weights

            error = target - prediction

            weights += eta * error * phi

        # Calculate training predictions after this epoch
        Phi = design_matrix(X, mus, sigma)
        predictions = Phi @ weights

        current_error = residual_error(
            predictions,
            targets
        )

        # Check convergence
        if abs(previous_error - current_error) < tolerance:
            stable_epochs += 1
        else:
            stable_epochs = 0

        if stable_epochs >= patience:
            return weights, epoch + 1

        previous_error = current_error

    return weights, max_epochs

def plot_rbf_results(x_test, targets, predictions, mus, title="RBF approximation"):
    plt.figure(figsize=(10, 5))

    # True function
    plt.plot(x_test, targets, label="Target function", linewidth=2)

    # RBF approximation
    plt.plot(x_test, predictions, "--", label="RBF prediction", linewidth=2)

    # RBF center positions
    plt.scatter(
        mus,
        np.zeros_like(mus),
        marker="x",
        s=70,
        label="RBF centers"
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
