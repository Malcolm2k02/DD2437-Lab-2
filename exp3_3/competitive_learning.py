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


def competitive_learning_dead_unit_test(
    X,
    n_units,
    eta=0.1,
    epochs=100,
    seed=42
):
    """
    Vanilla competitive learning with deliberately poor
    initialization to demonstrate dead units.
    """

    rng = np.random.default_rng(seed)

    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Deliberately initialize over a much wider interval
    # than the actual sine input data [0, 2*pi].
    centers = rng.uniform(
        -5,
        11,
        size=(n_units, X.shape[1])
    )

    wins = np.zeros(
        n_units,
        dtype=int
    )

    for epoch in range(epochs):

        order = rng.permutation(
            len(X)
        )

        for i in order:

            x = X[i]

            distances = np.linalg.norm(
                centers - x,
                axis=1
            )

            winner = np.argmin(
                distances
            )

            centers[winner] += (
                eta *
                (x - centers[winner])
            )

            wins[winner] += 1

    return centers, wins


def competitive_learning_multi_winner(
    X,
    n_units,
    eta=0.1,
    epochs=100,
    n_winners=2,
    seed=42
):
    """
    Competitive learning where the closest n_winners
    are updated instead of only the single closest unit.
    """

    rng = np.random.default_rng(seed)

    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # EXACTLY the same initialization as vanilla CL
    centers = rng.uniform(
        -5,
        11,
        size=(n_units, X.shape[1])
    )

    wins = np.zeros(
        n_units,
        dtype=int
    )

    for epoch in range(epochs):

        order = rng.permutation(
            len(X)
        )

        for i in order:

            x = X[i]

            distances = np.linalg.norm(
                centers - x,
                axis=1
            )

            # Find the n_winners closest units
            winners = np.argsort(
                distances
            )[:n_winners]

            # Closest winner gets full learning rate.
            # Second winner gets a smaller update.
            for rank, winner in enumerate(winners):

                winner_eta = (
                    eta / (rank + 1)
                )

                centers[winner] += (
                    winner_eta *
                    (x - centers[winner])
                )

                wins[winner] += 1

    return centers, wins