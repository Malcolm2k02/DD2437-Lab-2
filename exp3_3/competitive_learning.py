import numpy as np


def competitive_learning(X, n_units, eta=0.1, epochs=100, seed=42):

    rng = np.random.default_rng(seed)

    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Randomly choose training samples as initial centers
    indices = rng.choice(len(X), n_units, replace=False)
    centers = X[indices].copy()

    wins = np.zeros(n_units, dtype=int)

    for epoch in range(epochs):

        order = rng.permutation(len(X))

        for i in order:

            x = X[i]

            # Distance from x to every RBF center
            distances = np.linalg.norm(
                centers - x,
                axis=1
            )

            # Winner = closest RBF unit
            winner = np.argmin(distances)

            # Move winner towards x
            centers[winner] += eta * (
                x - centers[winner]
            )

            wins[winner] += 1

    return centers, wins


def calculate_sigmas(centers):

    centers = np.asarray(centers).flatten()

    sigmas = np.zeros(len(centers))

    for i in range(len(centers)):

        distances = np.abs(centers - centers[i])

        # Ignore distance to itself
        distances[i] = np.inf

        # Width based on closest neighbouring center
        sigmas[i] = np.min(distances)

    return sigmas

