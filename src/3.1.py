import numpy as np
import matplotlib.pyplot as plt
from rbf import gaussian_rbf, design_matrix, train_least_squares, predict_rbf, train_delta, residual_error, plot_rbf_results

x_train = np.arange(0, 2 * np.pi, 0.1)

x_test = np.arange(0.05, 2 * np.pi, 0.1)

sin_train = np.sin(2 * x_train)
sin_test = np.sin(2 * x_test)

square_train = np.where(np.sin(2 * x_train) >= 0, 1, -1)
square_test = np.where(np.sin(2 * x_test) >= 0, 1, -1)

for n_rbf in [8, 9, 12, 20]:
    mus = np.linspace(0, 2 * np.pi, n_rbf)
    sigma = 0.5

    Phi_train = design_matrix(x_train, mus, sigma)
    # print("Design matrix shape:", Phi_train.shape)
    weights = train_least_squares(Phi_train, sin_train)
    #print(weights)
    Phi_test = design_matrix(x_test, mus, sigma)
    predictions = Phi_test @ weights
    error = residual_error(predictions, sin_test)
    
    weights_square = train_least_squares(Phi_train, square_train)
    predictions_square = Phi_test @ weights_square
    transformed_square = np.where(predictions_square >= 0, 1, -1)
    error_square = residual_error(predictions_square, square_test)
    transformed_error_square = residual_error(transformed_square, square_test)
    print(f"RBFs: {n_rbf}, Sin error: {error:.6f}, Square error: {error_square:.6f}, Transformed square error: {transformed_error_square:.6f}")

    plot_rbf_results(x_test, sin_test, predictions, mus, title="RBF approximation of sin(2x)")
    plot_rbf_results(x_test, square_test, predictions_square, mus, title="RBF approximation of square wave")
    plot_rbf_results(x_test, square_test, transformed_square, mus, title="Transformed RBF approximation of square wave")

