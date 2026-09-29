"""Lab 2, task 4.3: unsupervised clustering of MPs by their 31 votes.

Run from the repository root: python -m exp4.task4_3
Only NumPy and Matplotlib are required. Outputs default to results/task4_3.
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Support both python -m exp4.task4_3 and python exp4/task4_3.py.
if __package__:
    from .task4_3_results import save_results
else:
    from task4_3_results import save_results


def load_data(directory):
    """Preserve row alignment and the supplied 0.5 missing-vote encoding."""
    votes = np.loadtxt(directory / "votes.dat", delimiter=",").reshape(349, 31)
    labels = {}
    for key, filename, upper in [("party", "mpparty.dat", 7),
                                  ("gender", "mpsex.dat", 1),
                                  ("district", "mpdistrict.dat", 29)]:
        values = np.loadtxt(directory / filename, comments="%")
        lower = 1 if key == "district" else 0
        if (values.shape != (349,) or not np.isfinite(values).all()
                or not np.equal(values, np.floor(values)).all()
                or np.any((values < lower) | (values > upper))):
            raise ValueError(f"Invalid {key} labels")
        labels[key] = values.astype(int)
    # The supplied names file is Latin-1 (Swedish accented characters).
    names = (directory / "mpnames.txt").read_text(encoding="latin-1").splitlines()
    if len(names) != 349 or not np.isin(votes, [0, 0.5, 1]).all():
        raise ValueError("Expected 349 names and votes encoded as 0, 0.5, or 1")
    return votes, labels, names


def winners(votes, weights):
    pos = np.zeros(len(votes), dtype=int)
    for j in range(len(votes)):
        distances = np.sum((weights - votes[j]) ** 2, axis=1)
        pos[j] = np.argmin(distances)
    return pos


def neighbourhood_mask(grid, winner, radius):
    return np.max(np.abs(grid - grid[winner]), axis=1) <= radius


def train_som(votes, epochs=50, seed=42, eta=0.2):
    """10x10 planar SOM, Euclidean BMUs and shrinking square neighbourhoods.

    The Chebyshev grid radius decreases from 5 to 0. Boundary neighbours
    are clipped naturally by grid distance; opposite edges do not wrap.
    The learning rate decreases linearly from eta to eta/10.
    """
    if epochs < 2 or not 0 < eta <= 1:
        raise ValueError("Use at least two epochs and 0 < eta <= 1")
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0, 1, (100, votes.shape[1]))
    grid = np.array([(row, column) for row in range(10) for column in range(10)])
    history = []
    for i in range(epochs):
        progress = i / (epochs - 1)
        # Rounded radius gives winner-only fine tuning in the final epochs.
        neighbourhood = int(round(5 * (1 - progress)))
        current_eta = eta * (1 - 0.9 * progress)
        for j in rng.permutation(len(votes)):
            vote = votes[j]
            distances = np.sum((weights - vote) ** 2, axis=1)
            winner = np.argmin(distances)

            neighbours = neighbourhood_mask(grid, winner, neighbourhood)
            weights[neighbours] += current_eta * (vote - weights[neighbours])
        bmu = winners(votes, weights)
        history.append(float(np.linalg.norm(votes - weights[bmu], axis=1).mean()))
    return weights, winners(votes, weights), np.array(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/task4_3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    votes, labels, names = load_data(args.data_dir)
    weights, bmu, history = train_som(votes, args.epochs, args.seed)
    save_results(votes, labels, names, weights, bmu, history,
                 args.output_dir, args.seed)


if __name__ == "__main__":
    main()
