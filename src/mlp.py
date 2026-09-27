import numpy as np


class MLP:
    def __init__(
        self,
        n_hidden,
        learning_rate=0.01,
        alpha=0.9,
        seed=42
    ):

        rng = np.random.default_rng(seed)

        # =====================================================
        # Weights and biases
        # =====================================================

        # Input -> hidden
        self.W1 = rng.normal(
            0,
            1.0,
            size=(1, n_hidden)
        )

        self.b1 = np.zeros(n_hidden)

        # Hidden -> output
        self.W2 = rng.normal(
            0,
            1.0,
            size=n_hidden
        )

        self.b2 = 0.0

        self.learning_rate = learning_rate
        self.alpha = alpha


        # =====================================================
        # Momentum terms
        # =====================================================

        self.dW1_old = np.zeros_like(self.W1)
        self.db1_old = np.zeros_like(self.b1)

        self.dW2_old = np.zeros_like(self.W2)
        self.db2_old = 0.0


    def forward(self, X):

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

            # Hidden -> output gradients
            grad_W2 = hidden.T @ error / n_samples
            grad_b2 = np.mean(error)


            # Propagate output error to hidden layer
            hidden_error = np.outer(
                error,
                self.W2
            )

            # tanh derivative
            hidden_delta = (
                hidden_error *
                (1 - hidden ** 2)
            )


            # Input -> hidden gradients
            grad_W1 = X.T @ hidden_delta / n_samples

            grad_b1 = np.mean(
                hidden_delta,
                axis=0
            )


            # =================================================
            # Momentum updates
            # =================================================

            dW2 = (
                self.alpha * self.dW2_old
                - self.learning_rate * grad_W2
            )

            db2 = (
                self.alpha * self.db2_old
                - self.learning_rate * grad_b2
            )

            dW1 = (
                self.alpha * self.dW1_old
                - self.learning_rate * grad_W1
            )

            db1 = (
                self.alpha * self.db1_old
                - self.learning_rate * grad_b1
            )


            # =================================================
            # Update weights
            # =================================================

            self.W2 += dW2
            self.b2 += db2

            self.W1 += dW1
            self.b1 += db1


            # Store updates for next epoch
            self.dW2_old = dW2
            self.db2_old = db2

            self.dW1_old = dW1
            self.db1_old = db1


        return errors


    def predict(self, X):

        predictions, _ = self.forward(X)

        return predictions


    def normalize_input(self, X):

        X = np.asarray(X).reshape(-1, 1)

        return (X - np.pi) / np.pi