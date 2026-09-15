import numpy as np
import matplotlib.pyplot as plt
from src.rbf import *
from src.data_setup import *

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

