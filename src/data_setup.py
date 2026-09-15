import numpy as np

np.random.seed(42)

# Inputs
x_train = np.arange(0, 2 * np.pi, 0.1)
x_test = np.arange(0.05, 2 * np.pi, 0.1)

# Clean functions
sin_train = np.sin(2 * x_train)
sin_test = np.sin(2 * x_test)

square_train = np.where(
    np.sin(2 * x_train) >= 0,
    1,
    -1
)

square_test = np.where(
    np.sin(2 * x_test) >= 0,
    1,
    -1
)

# Noise
noise_var = 0.1
noise_std = np.sqrt(noise_var)

sin_train_noisy = (
    sin_train
    + np.random.normal(
        0,
        noise_std,
        size=sin_train.shape
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

square_train_noisy = (
    square_train
    + np.random.normal(
        0,
        noise_std,
        size=square_train.shape
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