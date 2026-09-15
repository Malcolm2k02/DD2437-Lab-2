from src.data_setup import *
from src.rbf import *
import time

from src.data_setup import *
from src.rbf import *
import time

# ============================================================
# Configuration
# ============================================================

n_rbf = 50
sigma = 0.05
eta = 0.05
max_epochs = 3000

# Evenly spaced RBF centers
mus = np.linspace(0, 2 * np.pi, n_rbf)


# ============================================================
# Batch learning
# ============================================================

Phi_train = design_matrix(
    x_train,
    mus,
    sigma
)

start = time.perf_counter()

weights_batch = train_least_squares(
    Phi_train,
    square_train_noisy
)

batch_time = time.perf_counter() - start


# ============================================================
# Online delta learning
# ============================================================

start = time.perf_counter()

weights_online, epochs_trained = train_delta(
    x_train,
    square_train_noisy,
    mus,
    sigma,
    eta,
    max_epochs,
    tolerance=1e-3,
    patience=5
)

online_time = time.perf_counter() - start


# ============================================================
# Predictions
# ============================================================

Phi_test = design_matrix(
    x_test,
    mus,
    sigma
)

pred_batch = Phi_test @ weights_batch
pred_online = Phi_test @ weights_online


# ============================================================
# Raw regression errors
# ============================================================

batch_noisy_error = residual_error(
    pred_batch,
    square_test_noisy
)

online_noisy_error = residual_error(
    pred_online,
    square_test_noisy
)

batch_clean_error = residual_error(
    pred_batch,
    square_test
)

online_clean_error = residual_error(
    pred_online,
    square_test
)


# ============================================================
# Threshold predictions
# ============================================================

batch_thresholded = np.where(pred_batch >= 0, 1, -1)
online_thresholded = np.where(pred_online >= 0, 1, -1)

batch_threshold_error = residual_error(
    batch_thresholded,
    square_test
)

online_threshold_error = residual_error(
    online_thresholded,
    square_test
)


# ============================================================
# Results
# ============================================================

print("\n========== NOISY SQUARE WAVE ==========")

print("\nBatch least squares:")
print(f"Noisy test MAE: {batch_noisy_error:.5f}")
print(f"Clean test MAE: {batch_clean_error:.5f}")
print(f"Thresholded clean MAE: {batch_threshold_error:.5f}")
print(f"Training time: {batch_time:.6f} seconds")

print("\nOnline delta rule:")
print(f"Noisy test MAE: {online_noisy_error:.5f}")
print(f"Clean test MAE: {online_clean_error:.5f}")
print(f"Thresholded clean MAE: {online_threshold_error:.5f}")
print(f"Training time: {online_time:.6f} seconds")
print(f"Epochs trained: {epochs_trained}")

"""========== NOISY SQUARE WAVE ==========
with sigma = 1.0:

Batch least squares:
Noisy test MAE: 0.34100
Clean test MAE: 0.23403
Thresholded clean MAE: 0.03175
Training time: 0.036002 seconds

Online delta rule:
Noisy test MAE: 0.52983
Clean test MAE: 0.42348
Thresholded clean MAE: 0.12698
Training time: 105.553572 seconds
Epochs trained: 3000

with sigma = 0.5:
========== NOISY SQUARE WAVE ==========

Batch least squares:
Noisy test MAE: 0.49039
Clean test MAE: 0.38943
Thresholded clean MAE: 0.09524
Training time: 0.042803 seconds

Online delta rule:
Noisy test MAE: 0.40220
Clean test MAE: 0.27087
Thresholded clean MAE: 0.06349
Training time: 106.771148 seconds
Epochs trained: 3000

with sigma= 0.1:
    ========== NOISY SQUARE WAVE ==========

Batch least squares:
Noisy test MAE: 0.33443
Clean test MAE: 0.24485
Thresholded clean MAE: 0.06349
Training time: 0.084857 seconds

Online delta rule:
Noisy test MAE: 0.32889
Clean test MAE: 0.18307
Thresholded clean MAE: 0.06349
Training time: 1.128075 seconds
Epochs trained: 25

with sigma= 0.25: 
========== NOISY SQUARE WAVE ==========

Batch least squares:
Noisy test MAE: 0.44763
Clean test MAE: 0.34774
Thresholded clean MAE: 0.06349
Training time: 0.034361 seconds

Online delta rule:
Noisy test MAE: 0.35426
Clean test MAE: 0.20288
Thresholded clean MAE: 0.03175
Training time: 13.261017 seconds
Epochs trained: 304

with sigma= 0.05
========== NOISY SQUARE WAVE ==========

Batch least squares:
Noisy test MAE: 0.32444
Clean test MAE: 0.26266
Thresholded clean MAE: 0.03175
Training time: 0.083236 seconds

Online delta rule:
Noisy test MAE: 0.30954
Clean test MAE: 0.25334
Thresholded clean MAE: 0.03175
Training time: 2.323617 seconds
Epochs trained: 52"""


"""The RBF width controls the trade-off between locality and smoothness. 
Large widths produce strongly overlapping basis functions and smoother approximations, 
whereas smaller widths provide more localized responses that can better represent abrupt 
transitions such as those in the square wave. However, widths that are too small can reduce 
generalization between RBF centers."""
