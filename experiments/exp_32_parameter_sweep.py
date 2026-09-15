import numpy as np
from src.rbf import *
from src.data_setup import *


n_rbfs = [10, 15, 20, 25, 30, 35, 40, 45, 50]

sigmas = [
    0.1,
    0.25,
    0.5,
    1.0,
    1.5
]

etas = [
    0.001,
    0.01,
    0.05,
    0.1,
    0.5
]

max_epochs = 3000

results = []


for sigma in sigmas:

    for eta in etas:

        for n_rbf in n_rbfs:

            mus = np.linspace(
                0,
                2 * np.pi,
                n_rbf
            )

            weights, epochs = train_delta(
                x_train,
                sin_train_noisy,
                mus,
                sigma,
                eta,
                max_epochs
            )

            Phi_test = design_matrix(
                x_test,
                mus,
                sigma
            )

            predictions = Phi_test @ weights

            noisy_error = residual_error(
                predictions,
                sin_test_noisy
            )

            clean_error = residual_error(
                predictions,
                sin_test
            )

            results.append({
                "n_rbf": n_rbf,
                "sigma": sigma,
                "eta": eta,
                "epochs": epochs,
                "noisy_error": noisy_error,
                "clean_error": clean_error
            })


best = min(
    results,
    key=lambda r: r["noisy_error"]
)


print("\nBest configuration:")
print(best)