import numpy as np

np.random.seed(42)


# ============================================================
# Original input data
# ============================================================

# Original training samples
x_all_train = np.arange(0, 2 * np.pi, 0.1)

# Original shifted test samples -- keep these completely untouched
x_test = np.arange(0.05, 2 * np.pi, 0.1)


# ============================================================
# Split original training data into training + validation
# ============================================================

# Use 80% for training and 20% for validation
indices = np.arange(len(x_all_train))
np.random.shuffle(indices)

split = int(0.8 * len(indices))

train_indices = indices[:split]
val_indices = indices[split:]

# Sort so plotting remains in x-order
train_indices = np.sort(train_indices)
val_indices = np.sort(val_indices)

x_train = x_all_train[train_indices]
x_val = x_all_train[val_indices]


# ============================================================
# Clean functions
# ============================================================

# ----- Sine -----

sin_all_train = np.sin(2 * x_all_train)
sin_test = np.sin(2 * x_test)

sin_train = sin_all_train[train_indices]
sin_val = sin_all_train[val_indices]


# ----- Square -----

square_all_train = np.where(
    np.sin(2 * x_all_train) >= 0,
    1,
    -1
)

square_test = np.where(
    np.sin(2 * x_test) >= 0,
    1,
    -1
)

square_train = square_all_train[train_indices]
square_val = square_all_train[val_indices]


# ============================================================
# Add Gaussian noise
# ============================================================

noise_var = 0.1
noise_std = np.sqrt(noise_var)


# ----- Noisy sine -----

sin_train_noisy = (
    sin_train
    + np.random.normal(
        0,
        noise_std,
        size=sin_train.shape
    )
)

sin_val_noisy = (
    sin_val
    + np.random.normal(
        0,
        noise_std,
        size=sin_val.shape
    )
)

sin_test_noisy = (
    sin_test
    + np.random.normal(
        0,
        noise_std,
        size=sin_test.shape
    )
)


# ----- Noisy square -----

square_train_noisy = (
    square_train
    + np.random.normal(
        0,
        noise_std,
        size=square_train.shape
    )
)

square_val_noisy = (
    square_val
    + np.random.normal(
        0,
        noise_std,
        size=square_val.shape
    )
)

square_test_noisy = (
    square_test
    + np.random.normal(
        0,
        noise_std,
        size=square_test.shape
    )
)


# ============================================================
# Information
# ============================================================


# print("Dataset sizes:")
# print(f"Training:   {len(x_train)}")
# print(f"Validation: {len(x_val)}")
# print(f"Test:       {len(x_test)}")