import numpy as np


class MLP:
    def __init__(self, n_hidden, learning_rate=0.01, seed=42):

        rng = np.random.default_rng(seed)

        # Weights: input -> hidden layer
        self.W1 = rng.normal(
            0,
            1.0,
            size=(1, n_hidden)
        )

        # Biases for hidden layer
        self.b1 = np.zeros(n_hidden)

        # Weights: hidden -> output
        self.W2 = rng.normal(
            0,
            1.0,
            size=n_hidden
        )

        # Output bias
        self.b2 = 0.0

        self.learning_rate = learning_rate


    def forward(self, X):

        # Make X shape: (number of samples, 1)
        X = self.normalize_input(X)

        # Hidden layer
        z1 = X @ self.W1 + self.b1

        hidden = np.tanh(z1)

        # Linear output layer
        output = hidden @ self.W2 + self.b2

        return output, hidden


    def train(self, X, targets, epochs=3000):

        X = self.normalize_input(X)
        targets = np.asarray(targets)

        n_samples = len(X)

        errors = []

        for epoch in range(epochs):

            # =================================================
            # Forward pass
            # =================================================

            z1 = X @ self.W1 + self.b1

            hidden = np.tanh(z1)

            predictions = hidden @ self.W2 + self.b2


            # =================================================
            # Error
            # =================================================

            error = predictions - targets

            mse = np.mean(error ** 2)

            errors.append(mse)


            # =================================================
            # Backpropagation
            # =================================================

            # Output layer gradients
            dW2 = hidden.T @ error / n_samples

            db2 = np.mean(error)


            # Propagate error backwards through hidden layer
            hidden_error = np.outer(
                error,
                self.W2
            )

            # Derivative of tanh
            hidden_delta = (
                hidden_error *
                (1 - hidden ** 2)
            )


            # Input -> hidden gradients
            dW1 = X.T @ hidden_delta / n_samples

            db1 = np.mean(
                hidden_delta,
                axis=0
            )


            # =================================================
            # Update weights
            # =================================================

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1


        return errors


    def predict(self, X):

        predictions, _ = self.forward(X)

        return predictions

    def normalize_input(self, X):
        X = np.asarray(X).reshape(-1, 1)

        return (X - np.pi) / np.pi