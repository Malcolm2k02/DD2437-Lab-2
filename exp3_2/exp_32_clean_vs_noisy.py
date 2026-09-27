import numpy as np
from src.rbf import *
from src.data_setup import *


n_rbf = 50
sigma = 1.0
eta = 0.05
max_epochs = 3000

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


print(f"Noisy error: {noisy_error:.5f}")
print(f"Clean error: {clean_error:.5f}")
print(f"Epochs: {epochs}")