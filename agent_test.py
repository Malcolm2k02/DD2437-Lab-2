"""Small test program: Gaussian RBF evaluation with NumPy."""

import numpy as np


def gaussian(x, mu, sigma):
    """Compute the Gaussian RBF value for input x, center mu, width sigma."""
    return np.exp(-np.linalg.norm(x - mu) ** 2 / (2 * sigma ** 2))


if __name__ == "__main__":
    result = gaussian(x=1.0, mu=0.0, sigma=1.0)
    print(f"gaussian(x=1.0, mu=0.0, sigma=1.0) = {result}")
