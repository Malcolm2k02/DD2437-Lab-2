from src.data_setup import *
from src.rbf import *
import time

n_rbf = 50
sigma = 1.0
eta = 0.05
mus = np.linspace(0, 2 * np.pi, n_rbf)

# Batch
Phi_train = design_matrix(
    x_train,
    mus,
    sigma
)
start = time.perf_counter()

weights_batch = train_least_squares(
    Phi_train,
    sin_train_noisy
)

batch_time = time.perf_counter() - start


# Online
start = time.perf_counter()
weights_online, epochs_trained = train_delta(
    x_train,
    sin_train_noisy,
    mus,
    sigma,
    eta,
    3000,
    tolerance=1e-3,
    patience=5
)

online_time = time.perf_counter() - start

Phi_test = design_matrix(
    x_test,
    mus,
    sigma
)

pred_batch = Phi_test @ weights_batch
pred_online = Phi_test @ weights_online

batch_noisy_error = residual_error(
    pred_batch,
    sin_test_noisy
)

online_noisy_error = residual_error(
    pred_online,
    sin_test_noisy
)

batch_clean_error = residual_error(
    pred_batch,
    sin_test
)

online_clean_error = residual_error(
    pred_online,
    sin_test
)


print("\n========== BATCH VS ONLINE ==========")

print("\nBatch least squares:")
print(f"Noisy test MAE: {batch_noisy_error:.5f}")
print(f"Clean test MAE: {batch_clean_error:.5f}")
print(f"Training time: {batch_time:.6f} seconds")

print("\nOnline delta rule:")
print(f"Noisy test MAE: {online_noisy_error:.5f}")
print(f"Clean test MAE: {online_clean_error:.5f}")
print(f"Training time: {online_time:.6f} seconds")
print(f"Epochs trained: {epochs_trained}")

"""========== BATCH VS ONLINE ==========

Batch least squares:
Noisy test MAE: 0.27730
Clean test MAE: 0.14774
Training time: 0.016474 seconds

Online delta rule:
Noisy test MAE: 0.26975
Clean test MAE: 0.11142
Training time: 115.161265 seconds
Epochs trained: 3000"""